from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.article_pull import extract_article_row
from src.data.article_pull import pull_articles
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


def test_pull_articles_skips_existing_row_by_article_id(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "nyt.tsv"
    checkpoint_path = tmp_path / "checkpoint.json"
    output_path.write_text(
        (
            "article_id\tsource\tquery\tarticle_day\tpublished_at\theadline\tarticle_text\t"
            "snippet\tabstract\tlead_paragraph\tsection_name\tsubsection_name\tweb_url\tkeywords\n"
            "nyt://article/1\tnew_york_times\t\t2023-01-10\t2023-01-10T08:30:00+0000\tHeadline\t"
            "Lead paragraph\tSnippet\tAbstract\tLead paragraph\tBusiness\t\thttps://example.com/1\t"
            "economy | stocks\n"
        ),
        encoding="utf-8",
    )

    def fake_fetch(**kwargs):
        return {
            "response": {
                "docs": [
                    {
                        "_id": "nyt://article/1",
                        "pub_date": "2023-01-10T08:30:00+0000",
                        "headline": {"main": "Headline"},
                        "snippet": "Snippet",
                        "abstract": "Abstract",
                        "lead_paragraph": "Lead paragraph",
                        "section_name": "Business",
                        "web_url": "https://example.com/1",
                        "keywords": [{"value": "economy"}, {"value": "stocks"}],
                    }
                ],
                "meta": {"hits": 1},
            }
        }

    monkeypatch.setattr("src.data.article_pull.fetch_nyt_archive_month", fake_fetch)

    rows_written = pull_articles(
        api_key="dummy",
        query="",
        start_date="2023-01-01",
        end_date="2023-01-31",
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        mode="archive",
    )

    assert rows_written == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_pull_articles_skips_existing_row_without_article_id_by_web_url(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "nyt.tsv"
    checkpoint_path = tmp_path / "checkpoint.json"
    output_path.write_text(
        (
            "article_id\tsource\tquery\tarticle_day\tpublished_at\theadline\tarticle_text\t"
            "snippet\tabstract\tlead_paragraph\tsection_name\tsubsection_name\tweb_url\tkeywords\n"
            "\tnew_york_times\t\t2023-01-10\t2023-01-10T08:30:00+0000\tHeadline\t"
            "Lead paragraph\tSnippet\tAbstract\tLead paragraph\tBusiness\t\thttps://example.com/1\t"
            "economy | stocks\n"
        ),
        encoding="utf-8",
    )

    def fake_fetch(**kwargs):
        return {
            "response": {
                "docs": [
                    {
                        "_id": "",
                        "pub_date": "2023-01-10T08:30:00+0000",
                        "headline": {"main": "Headline"},
                        "snippet": "Snippet",
                        "abstract": "Abstract",
                        "lead_paragraph": "Lead paragraph",
                        "section_name": "Business",
                        "web_url": "https://example.com/1",
                        "keywords": [{"value": "economy"}, {"value": "stocks"}],
                    }
                ],
                "meta": {"hits": 1},
            }
        }

    monkeypatch.setattr("src.data.article_pull.fetch_nyt_archive_month", fake_fetch)

    rows_written = pull_articles(
        api_key="dummy",
        query="",
        start_date="2023-01-01",
        end_date="2023-01-31",
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        mode="archive",
    )

    assert rows_written == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_pull_articles_ignores_mismatched_checkpoint_and_fetches_earlier_dates(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "nyt.tsv"
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        (
            "{\n"
            '  "current_date": "2015-01-10",\n'
            '  "next_page": 3,\n'
            '  "mode": "search",\n'
            '  "query": "",\n'
            '  "start_date": "2015-01-10",\n'
            '  "end_date": "2015-01-31",\n'
            f'  "output_path": "{output_path}",\n'
            '  "updated_at": "2026-01-01T00:00:00Z"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    called_begin_dates: list[str] = []

    def fake_search(**kwargs):
        called_begin_dates.append(kwargs["begin_date"])
        return {"response": {"docs": [], "meta": {"hits": 0}}}

    monkeypatch.setattr("src.data.article_pull.search_nyt_articles", fake_search)

    rows_written = pull_articles(
        api_key="dummy",
        query="",
        start_date="2015-01-01",
        end_date="2015-01-01",
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        mode="search",
        page_limit=1,
    )

    assert rows_written == 0
    assert called_begin_dates == ["20150101"]
