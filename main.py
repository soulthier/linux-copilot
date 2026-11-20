import json
import os
import requests

MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
RAW_PATH = "data/raw/enterprise-attack.json"
PROCESSED_PATH = "data/processed/mitre_docs.json"


def download_mitre() -> dict:
    if os.path.exists(RAW_PATH):
        print(f"Файл уже скачан: {RAW_PATH}")
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Скачиваем базу MITRE ATT&CK (около 35-45 МБ)...")
    response = requests.get(MITRE_URL, timeout=60)
    response.raise_for_status()
    data = response.json()

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Сохранено в {RAW_PATH}")
    return data


def parse_mitre(stix_bundle: dict) -> list[dict]:
    objects = stix_bundle.get("objects", [])

    techniques = {}
    mitigations = {}
    mitigates_rels = []

    for obj in objects:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        obj_type = obj.get("type")
        obj_id = obj.get("id")

        if obj_type == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            mitre_id = next(
                (r["external_id"] for r in ext_refs if r.get("source_name") == "mitre-attack"),
                None,
            )
            if mitre_id:
                techniques[obj_id] = {
                    "mitre_id": mitre_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "platforms": obj.get("x_mitre_platforms", []),
                    "tactics": [
                        phase["phase_name"]
                        for phase in obj.get("kill_chain_phases", [])
                    ],
                }

        elif obj_type == "course-of-action":
            mitigations[obj_id] = {
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
            }

        elif obj_type == "relationship" and obj.get("relationship_type") == "mitigates":
            mitigates_rels.append(
                (obj.get("source_ref"), obj.get("target_ref"), obj.get("description", ""))
            )

    tech_mitigations = {tech_id: [] for tech_id in techniques}
    for source_ref, target_ref, rel_desc in mitigates_rels:
        if source_ref in mitigations and target_ref in techniques:
            mit = mitigations[source_ref]
            tech_mitigations[target_ref].append(
                f"- {mit['name']}: {rel_desc or mit['description']}"
            )

    documents = []
    for tech_id, tech in techniques.items():
        mits = tech_mitigations.get(tech_id, [])
        mitigation_text = "\n".join(mits) if mits else "No specific MITRE mitigation listed."

        full_text = (
            f"Technique: {tech['mitre_id']} - {tech['name']}\n"
            f"Tactics: {', '.join(tech['tactics'])}\n"
            f"Platforms: {', '.join(tech['platforms'])}\n\n"
            f"Description:\n{tech['description']}\n\n"
            f"Mitigations:\n{mitigation_text}"
        )

        documents.append(
            {
                "doc_id": tech["mitre_id"],
                "title": f"{tech['mitre_id']}: {tech['name']}",
                "tactics": tech["tactics"],
                "platforms": tech["platforms"],
                "text": full_text,
            }
        )

    return documents


def main():
    bundle = download_mitre()
    docs = parse_mitre(bundle)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Готово! Обработано техник: {len(docs)}")
    print(f"Результат сохранён в: {PROCESSED_PATH}")
    if docs:
        print("\nПример первого документа:")
        print(docs[0]["text"][:500] + "...\n")


if __name__ == "__main__":
    main()