"""
strip_comments.py

Privremena, LOKALNA alatka - uklanja (ili vraca) komentare i docstring-ove
iz .py fajlova u trenutnom folderu.

POKRETANJE:
    python strip_comments.py strip     -> pravi backup i uklanja komentare
    python strip_comments.py restore   -> vraca originalne (komentarisane) fajlove

Backup se cuva JEDAN NIVO IZNAD trenutnog foldera (van git repozitorijuma),
u folderu "_komentari_backup" - zato ne mora da se pominje u .gitignore,
niti postoji rizik da se slucajno doda u git.

VAZNA NAPOMENA: posle "restore" komande, "git status" ce pokazati da su
fajlovi izmenjeni (jer se lokalno razlikuju od poslednjeg commit-a) - to
je OCEKIVANO i BEZOPASNO, samo ne pravi novi commit posle restore-a dok
prethodno ponovo ne pokrenes "strip".
"""

import ast
import io
import shutil
import sys
import tokenize
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BACKUP_DIR = THIS_DIR.parent / "_komentari_backup"
SELF_NAME = Path(__file__).name


def get_docstring_line_ranges(source: str):
    """Vraca skup brojeva linija (1-indeksirano) koje pripadaju
    docstring iskazima (modul/klasa/funkcija)."""
    tree = ast.parse(source)
    nodes = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    line_set = set()
    for node in nodes:
        if node.body:
            first = node.body[0]
            value = getattr(first, "value", None)
            if isinstance(first, ast.Expr) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                start = first.lineno
                end = getattr(first, "end_lineno", start)
                line_set.update(range(start, end + 1))
    return line_set


def get_comment_columns(source: str):
    """Vraca {broj_linije: kolona_pocetka_komentara}, koriscenjem
    tokenize modula (bezbedno - ne lomi # unutar string literala)."""
    comment_cols = {}
    readline = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT:
            line_no, col = tok.start
            comment_cols[line_no] = col
    return comment_cols


def clean_source(source: str) -> str:
    docstring_lines = get_docstring_line_ranges(source)
    comment_cols = get_comment_columns(source)

    lines = source.splitlines()
    out_lines = []
    for i, line in enumerate(lines, start=1):
        if i in docstring_lines:
            continue
        if i in comment_cols:
            line = line[: comment_cols[i]].rstrip()
            if line == "":
                continue
        out_lines.append(line)

    # najvise jedna prazna linija zaredom (citljivost)
    result, blank_run = [], 0
    for line in out_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        result.append(line)
    return "\n".join(result) + "\n"


def strip_all():
    BACKUP_DIR.mkdir(exist_ok=True)
    py_files = [f for f in THIS_DIR.glob("*.py") if f.name != SELF_NAME]

    if not py_files:
        print("Nema .py fajlova za obradu.")
        return

    for f in py_files:
        backup_path = BACKUP_DIR / f.name
        if backup_path.exists():
            print(f"[preskačem] {f.name} - backup već postoji (verovatno već stripovan)")
            continue
        source = f.read_text(encoding="utf-8")
        shutil.copy2(f, backup_path)
        try:
            cleaned = clean_source(source)
        except SyntaxError as e:
            print(f"[GREŠKA] {f.name}: ne mogu da parsiram ({e}) - preskačem, backup ostaje")
            continue
        f.write_text(cleaned, encoding="utf-8")
        print(f"[očišćeno] {f.name}  (backup: {backup_path})")

    print(f"\nGotovo. Originali su sačuvani u: {BACKUP_DIR}")


def restore_all():
    if not BACKUP_DIR.exists():
        print(f"Ne postoji backup folder ({BACKUP_DIR}) - nema šta da se vrati.")
        return

    backups = list(BACKUP_DIR.glob("*.py"))
    if not backups:
        print("Backup folder je prazan.")
        return

    for backup_path in backups:
        target = THIS_DIR / backup_path.name
        shutil.copy2(backup_path, target)
        print(f"[vraćeno] {target.name}")

    print(
        "\nGotovo - originalni (komentarisani) fajlovi su vraćeni.\n"
        "NAPOMENA: 'git status' će sada pokazati izmene u odnosu na poslednji "
        "commit - to je očekivano. Ne pravi novi commit dok ponovo ne pokreneš "
        "'strip' ako budeš hteo da ažuriraš git."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("strip", "restore"):
        print("Upotreba: python strip_comments.py [strip|restore]")
        sys.exit(1)

    if sys.argv[1] == "strip":
        strip_all()
    else:
        restore_all()
