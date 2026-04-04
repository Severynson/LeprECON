from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.article_pull import extract_article_row
from src.data.article_pull import load_env_value
from src.data.article_pull import normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("line 1\tline 2\nline 3") == "line 1 line 2 line 3"


def test_load_env_value_reads_dotenv(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("NYT_API_KEY='secret-key'\n", encoding="utf-8")
    assert load_env_value("NYT_API_KEY", env_path=str(env_path)) == "secret-key"


def test_extract_article_row_prefers_lead_paragraph() -> None:
    doc = {
        "_id": "nyt://article/123",
        "pub_date": "2023-01-10T08:30:00+0000",
        "headline": {"main": "Headline"},
        "snippet": "Snippet",
        "abstract": "Abstract",
        "lead_paragraph": "Lead paragraph",
        "section_name": "Business",
        "web_url": "https://example.com/article",
        "keywords": [{"value": "economy"}, {"value": "stocks"}],
    }

    row = extract_article_row(doc, "S&P 500")

    assert row["article_id"] == "nyt://article/123"
    assert row["article_day"] == "2023-01-10"
    assert row["article_text"] == "Lead paragraph"
    assert row["headline"] == "Headline"
    assert row["keywords"] == "economy | stocks"
