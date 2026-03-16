from concurrent.futures import ThreadPoolExecutor, as_completed
from main import fetch_clean, chunks, summarize_block


def summarize_url_parallel(
        url,
        model="phi3:mini",
        bullets=6,
        max_words=300,
        workers=2
):

    title, text = fetch_clean(url)

    # prevent huge pages
    text = text[:20000]

    text_chunks = list(chunks(text))

    sections = []

    # parallel inference
    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = [
            executor.submit(
                summarize_block,
                chunk,
                model,
                min(7, bullets),
                min(220, max_words)
            )
            for chunk in text_chunks
        ]

        for future in as_completed(futures):

            try:
                sections.append(future.result())
            except Exception:
                sections.append({
                    "abstract": "",
                    "bullets": []
                })

    # merge intermediate summaries
    merged_abs = " ".join(
        s.get("abstract", "") for s in sections
    )[:10000]

    merged_bul = [
        b for s in sections for b in s.get("bullets", [])
    ][:bullets]

    # final summarization pass
    final = summarize_block(
        merged_abs,
        model,
        bullets,
        max_words
    )

    final.setdefault("abstract", merged_abs[:max_words * 8])
    final["bullets"] = final.get("bullets", []) or merged_bul
    final["title"] = title
    final["url"] = url

    return final