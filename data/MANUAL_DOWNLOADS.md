# Manual downloads

`python scripts/download_corpus.py` fetches most of the corpus automatically. A few files
have to be saved by hand with a normal browser. This page tells you which and how.

## The three EUR-Lex files (always manual)

EUR-Lex, the EU's official law portal, protects its pages with a JavaScript check that a
script cannot pass (a browser passes it in a second or two). RegLens does not try to get
around it - you save these three PDFs yourself.

| Save as (inside `data/raw/`) | Open this page | Shortcut to the English PDF |
|---|---|---|
| `eu-gdpr.pdf` | https://eur-lex.europa.eu/eli/reg/2016/679/oj | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679 |
| `eu-ai-act.pdf` | https://eur-lex.europa.eu/eli/reg/2024/1689/oj | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 |
| `eu-ai-omnibus.pdf` | https://eur-lex.europa.eu/eli/reg/2026/1744/oj | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202601744 |

Steps (same for all three):

1. Open the "Shortcut" address in Chrome, Edge or Firefox. If it shows a web page instead of a PDF,
   open the first link and use the **PDF** download option offered on the page (choose English).
2. Make sure it is the **English** version of the **Official Journal** text.
3. Save it into the `data/raw/` folder of this project using the exact file name from the table.

Windows tip: if you cannot see file extensions, turn them on (File Explorer > View > Show >
File name extensions) so you do not end up with `eu-gdpr.pdf.pdf`.

Why two of these matter together: the AI Act (`eu-ai-act.pdf`) was amended by Regulation
(EU) 2026/1744 (`eu-ai-omnibus.pdf`), which moved the application dates of the high-risk
obligations. RegLens indexes both so it can show how an answer depends on which document
is newer.

## If another download fails

The downloader tells you which files it could not fetch, and for each one prints the page to
open and the exact file name to use. The same information is in `data/sources.csv`
(columns `landing_url` and `filename`).

- **PDF sources:** open the address in your browser and save the PDF into `data/raw/` under
  the exact file name.
- **Web-page sources** (file names ending in `.html`): open the page, then use the browser's
  *Save page as* and choose **Webpage, HTML only**. Name it exactly as listed.

## Check your work

```
python scripts/download_corpus.py --status
```

This makes no network requests. It lists which files are present and valid, which are missing,
and any file in `data/raw/` whose name it does not recognise (for example `eu-gdpr (1).pdf`).
You need at least 20 usable documents; the manifest has 34, so a few failures are fine.
