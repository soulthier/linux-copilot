import json
import os
import requests

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RAW_PATH = "data/raw/nvd_cves.json"
PROCESSED_PATH = "data/processed/cve_docs.json"


def fetch_recent_cves(results_per_page: int = 200) -> dict:
    if os.path.exists(RAW_PATH):
        print(f"Файл уже скачан: {RAW_PATH}")
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Запрашиваем последние {results_per_page} CVE из NVD API...")
    params = {"resultsPerPage": results_per_page}
    response = requests.get(NVD_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Сохранено в {RAW_PATH}")
    return data


def extract_cvss(metrics: dict) -> tuple[str, float | None]:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        items = metrics.get(key)
        if items:
            cvss_data = items[0].get("cvssData", {})
            severity = cvss_data.get("baseSeverity") or items[0].get("baseSeverity") or "UNKNOWN"
            score = cvss_data.get("baseScore")
            return severity.upper(), score
    return "UNKNOWN", None


def parse_cves(nvd_data: dict) -> list[dict]:
    vulns = nvd_data.get("vulnerabilities", [])
    docs = []

    for item in vulns:
        cve = item.get("cve", {})
        cve_id = cve.get("id", "")
        descriptions = cve.get("descriptions", [])
        en_desc = next(
            (d["value"] for d in descriptions if d.get("lang") == "en"),
            descriptions[0]["value"] if descriptions else "",
        )

        severity, score = extract_cvss(cve.get("metrics", {}))

        # Извлекаем CWE (тип уязвимости), если есть
        cwes = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                val = desc.get("value", "")
                if val.startswith("CWE-"):
                    cwes.append(val)

        text = (
            f"CVE ID: {cve_id}\n"
            f"Severity: {severity} (Score: {score if score is not None else 'N/A'})\n"
            f"Weaknesses: {', '.join(cwes) if cwes else 'N/A'}\n\n"
            f"Description:\n{en_desc}"
        )

        docs.append(
            {
                "doc_id": cve_id,
                "title": f"{cve_id} [{severity}]",
                "severity": severity,
                "score": score,
                "cwes": cwes,
                "text": text,
            }
        )

    return docs


def main():
    raw_data = fetch_recent_cves(results_per_page=200)
    docs = parse_cves(raw_data)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Готово! Обработано CVE: {len(docs)}")
    print(f"Результат сохранён в: {PROCESSED_PATH}")
    if docs:
        print("\nПример первой CVE:")
        print(docs[0]["text"][:500] + "\n")


if __name__ == "__main__":
    main()