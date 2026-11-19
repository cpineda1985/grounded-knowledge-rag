import os

from akr.cli import load_dotenv


def test_load_dotenv_reads_values_and_keeps_existing_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nAKR_TEST_A="one"\nexport AKR_TEST_B=two\nAKR_TEST_C=\nAKR_TEST_D=from-file\n')
    monkeypatch.delenv("AKR_TEST_A", raising=False)
    monkeypatch.delenv("AKR_TEST_B", raising=False)
    monkeypatch.delenv("AKR_TEST_C", raising=False)
    monkeypatch.setenv("AKR_TEST_D", "from-shell")

    load_dotenv(env)

    assert os.environ["AKR_TEST_A"] == "one"
    assert os.environ["AKR_TEST_B"] == "two"
    assert "AKR_TEST_C" not in os.environ  # empty values are skipped
    assert os.environ["AKR_TEST_D"] == "from-shell"  # the shell wins


def test_load_dotenv_ignores_missing_file(tmp_path):
    load_dotenv(tmp_path / "missing.env")
