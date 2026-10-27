"""Rich text (the HTML subset QTextBrowser shows) for the License and Credits dialogs."""

from html import escape

from anima_trainer.about.credits import Credit, CreditSection
from anima_trainer.about.license import COPYRIGHT, GPL_URL, LICENSE_ID, NOTICE_PARAGRAPHS


def _link(url: str, text: str) -> str:
    if not url:
        return escape(text)
    return f'<a href="{escape(url)}">{escape(text)}</a>'


def license_html(app_name: str, version: str) -> str:
    paragraphs = "".join(f"<p>{escape(text)}</p>" for text in NOTICE_PARAGRAPHS)
    return (
        f"<h2>{escape(app_name)} {escape(version)}</h2>"
        f"<p>{escape(COPYRIGHT)}<br>License: {escape(LICENSE_ID)}</p>"
        f"{paragraphs}"
        f"<p>{_link(GPL_URL, 'GNU General Public License, version 3')}</p>"
    )


def _credit_html(credit: Credit) -> str:
    licenses = ", ".join(_link(url, f"license {n}") for n, url in enumerate(credit.license_urls, 1))
    license_line = escape(credit.license) + (f" ({licenses})" if licenses else "")
    notes = "".join(f"<br><i>{escape(note)}</i>" for note in credit.notes)
    return (
        f"<p><b>{_link(credit.url, credit.name)}</b><br>{escape(credit.role)}"
        f"<br>License: {license_line}{notes}</p>"
    )


def credits_html(sections: tuple[CreditSection, ...]) -> str:
    return "".join(
        f"<h3>{escape(section.title)}</h3>" + "".join(_credit_html(c) for c in section.credits)
        for section in sections
    )
