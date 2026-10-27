from anima_trainer.about.credits import AUTHOR, CITRON, CREDIT_SECTIONS, MODELS, Credit
from anima_trainer.models.catalog import BASE_MODEL_URLS, SUPPORT_MODEL_URLS


def _all_credits() -> tuple[Credit, ...]:
    return tuple(credit for section in CREDIT_SECTIONS for credit in section.credits)


def test_sections_start_with_the_author_then_citron():
    first, second = CREDIT_SECTIONS[0].credits, CREDIT_SECTIONS[1].credits

    assert first == (AUTHOR,)
    assert CITRON in second
    assert AUTHOR.license == "GPL-3.0-or-later"
    assert CITRON.license == "MIT"
    assert "citronlegacy" in CITRON.url


def test_every_downloadable_model_has_a_credit():
    covered = {key for credit in MODELS for key in credit.model_keys}

    assert covered == {*BASE_MODEL_URLS, *SUPPORT_MODEL_URLS}


def test_anima_credit_names_the_non_commercial_and_nvidia_licenses():
    anima = next(credit for credit in MODELS if "anima-base-v1.0" in credit.model_keys)

    assert "Non-Commercial" in anima.license
    assert "NVIDIA Open Model License" in anima.license


def test_every_credit_has_a_name_role_and_license_and_https_links():
    for credit in _all_credits():
        assert credit.name and credit.role and credit.license
        assert credit.url == "" or credit.url.startswith("https://")
        assert all(url.startswith("https://") for url in credit.license_urls)


def test_section_titles_are_unique():
    titles = [section.title for section in CREDIT_SECTIONS]

    assert len(titles) == len(set(titles))


def test_anima_credit_role_names_every_base_model_of_the_catalog():
    anima = next(credit for credit in MODELS if "anima-base-v1.0" in credit.model_keys)

    for key in BASE_MODEL_URLS:
        assert key in anima.role
