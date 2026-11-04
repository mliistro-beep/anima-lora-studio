from anima_trainer.about.credits import Credit, CreditSection
from anima_trainer.about.license import COPYRIGHT, GPL_URL
from anima_trainer.about.render import credits_html, license_html


def test_license_html_shows_name_version_copyright_and_gpl_link():
    page = license_html("Anima LoRA Studio", "0.2.0")

    assert "Anima LoRA Studio 0.2.0" in page
    assert COPYRIGHT in page
    assert f'href="{GPL_URL}"' in page


def test_credits_html_lists_sections_links_and_notes():
    section = CreditSection(
        "Models",
        (
            Credit(
                "Anima",
                "Base model",
                "https://example.org/anima",
                "Some License",
                license_urls=("https://example.org/license",),
                notes=("Non-commercial use only.",),
            ),
        ),
    )

    page = credits_html((section,))

    assert "<h3>Models</h3>" in page
    assert '<a href="https://example.org/anima">Anima</a>' in page
    assert '<a href="https://example.org/license">' in page
    assert "Non-commercial use only." in page


def test_credits_html_escapes_text_and_omits_the_link_without_url():
    section = CreditSection("A & B", (Credit("<me>", "role", "", "MIT"),))

    page = credits_html((section,))

    assert "A &amp; B" in page
    assert "&lt;me&gt;" in page
    assert "<a " not in page
