# Workflows Aktivieren - Praktische Anleitung

## 🎯 Das Problem

Du siehst auf der Branch Protection Seite:
```
No required checks
No checks have been added
```

**Grund:** Die GitHub Actions Workflows müssen erst einmal laufen, bevor sie als Status Checks verfügbar sind.

## 🚀 Schnelle Lösung (5 Schritte)

### Schritt 1: Workflows zu GitHub pushen

```bash
cd /Users/bartz/workspace/spotforecast2

# Sicherstellen, dass du auf main bist
git checkout main
git pull origin main

# Alle Workflow-Dateien hinzufügen
git add .github/workflows/
git add .releaserc.json
git add RELEASE_MANAGEMENT.md
git add GITHUB_DESKTOP_GUIDE.md
git add BRANCH_PROTECTION_GUIDE.md

# Committen
git commit -m "ci: vollautomatisierte Release-Pipeline eingerichtet"

# Pushen
git push origin main
```

### Schritt 2: Actions Tab öffnen

1. Gehe zu: https://github.com/sequential-parameter-optimization/spotforecast2
2. Klicke oben auf: **`Actions`** Tab
3. Links siehst du: `CI Tests`, `Release`, `Documentation`

### Schritt 3: Workflows aktivieren (falls nötig)

Falls die Workflows grau sind und deaktiviert:
1. Klicke `Enable on this repository`
2. Bestätige

Falls bereits aktiv (grüne Haken): Weiter zu Schritt 4.

### Schritt 4: Workflows triggern

Die Workflows sollten automatisch nach dem Push laufen.

**Wenn sie nicht starten:**
1. Erstelle einen kleinen Test-Push:
   ```bash
   git commit -m "chore: trigger workflows" --allow-empty
   git push origin main
   ```

2. Oder erstelle einen Pull Request:
   ```bash
   git checkout -b setup/trigger-workflows
   git commit -m "chore: setup" --allow-empty
   git push origin setup/trigger-workflows
   # Dann auf GitHub: "Create Pull Request"
   ```

### Schritt 5: Warten & überprüfen

1. Gehe zu Actions Tab
2. Du siehst die laufenden Workflows
3. **Warte 5-10 Minuten** bis alle durchgelaufen sind
4. Alle sollten 🟢 **grün** sein

## ✅ Jetzt können Status Checks hinzugefügt werden!

Nach der ersten erfolgreichen Workflow-Ausführung:

1. Gehe zu: https://github.com/sequential-parameter-optimization/spotforecast2/settings/branch_protection_rules/new
2. Branch name: `main`
3. ☑️ Require a pull request before merging
4. ☑️ Require status checks to pass before merging
5. **Scrolle nach unten** → Jetzt sind die Tests sichtbar!
   ```
   ☑️ Test on Python 3.13
   ☑️ Test on Python 3.12
   ☑️ Code Quality
   ☑️ Security Scan
   ```
6. Aktiviere mindestens `Test on Python 3.13`
7. Klicke `Create` (grüner Button unten rechts)

## 🔍 Debugging: Warum laufen die Workflows nicht?

### Problem 1: Workflows sind in `.github/workflows/` aber nicht in GitHub sichtbar

**Lösung:**
```bash
# Stelle sicher, dass alle Dateien gepusht wurden
git log --oneline | head -5
# Sollte deinen commit mit "ci:" zeigen

# Überprüfe Remote
git ls-remote origin | grep refs/heads/main
```

### Problem 2: Actions Tab zeigt die Workflows nicht

**Lösung:**
1. Repository Einstellungen prüfen:
   https://github.com/sequential-parameter-optimization/spotforecast2/settings/actions
2. Suche: "Actions permissions"
3. Stelle sicher: "Allow all actions and reusable workflows" ist ausgewählt

### Problem 3: Workflows starten aber schlagen fehl

**Häufige Fehler:**
- `pytest` nicht installiert → Siehe `.github/workflows/ci.yml` Zeile ~20
- `dependencies` Fehler → Siehe Job-Logs
- `pyproject.toml` nicht gefunden → Muss im Root sein

**Logs anschauen:**
1. Actions Tab
2. Klicke auf fehlgeschlagenen Workflow
3. Klicke auf Job (z.B. "Test on Python 3.13")
4. Scrolle durch Log um Fehler zu sehen

## 📋 Checkliste

- [ ] `.github/workflows/ci.yml` existiert
- [ ] `.github/workflows/release.yml` existiert
- [ ] `.github/workflows/docs.yml` existiert
- [ ] `.github/workflows/*.yml` wurden zu main gepusht
- [ ] Actions Tab zeigt laufende Workflows
- [ ] Nach 5-10 Min: Alle Workflows sind 🟢 grün
- [ ] Status Checks in Branch Protection sind jetzt sichtbar
- [ ] `Test on Python 3.13` wurde ausgewählt
- [ ] Branch Protection Rule erstellt

## 🎉 Nächste Schritte

Nach erfolgreicher Workflow-Aktivierung:

1. **Branch schützen** (siehe [BRANCH_PROTECTION_GUIDE.md](BRANCH_PROTECTION_GUIDE.md))
2. **Ersten PR erstellen** als Test
3. **Tests beobachten** (sollten automatisch laufen)
4. **Merge und Release** (automatisch)

## 🆘 Noch Probleme?

Wenn die Workflows immer noch nicht laufen:

1. Prüfe die `.github/workflows/ci.yml` Datei
2. Stelle sicher die Syntax ist korrekt (YAML Format)
3. Prüfe: Sind alle Dependencies verfügbar?
4. Lösche einen Workflow und erstelle ihn neu

Oder teste lokal:
```bash
# Lokal pytest ausführen
cd /Users/bartz/workspace/spotforecast2
pytest tests/ -v
```

## 📺 Video-Tutorial (Optional)

"GitHub Actions setup for Python" auf YouTube für visuelle Anleitung.

## 💡 Kurz: So funktioniert es

```
1. Code pushen
   ↓
2. GitHub Actions triggered automatisch
   ↓
3. Workflows in `.github/workflows/` laufen
   ↓
4. Status Checks werden erzeugt
   ↓
5. Status Checks erscheinen in Branch Protection
   ↓
6. Branch kann geschützt werden
```
