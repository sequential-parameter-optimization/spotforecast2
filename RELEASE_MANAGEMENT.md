# Release Management Strategie für spotforecast2

## 🎯 Übersicht

Diese vollautomatisierte Release-Management-Strategie ermöglicht sichere, professionelle Releases ohne manuelle Eingriffe und ohne Kosten.

> **💡 GitHub Desktop Nutzer:** Siehe [GITHUB_DESKTOP_GUIDE.md](GITHUB_DESKTOP_GUIDE.md) für eine detaillierte Anleitung mit der grafischen Oberfläche!

## 🚀 Workflow

### Automatische Releases

Releases werden **vollautomatisch** erstellt, wenn Änderungen auf den `main` Branch gepusht werden.

```
Entwicklung → Pull Request → Tests → Merge zu main → Automatisches Release
```

### Was passiert automatisch?

1. **Tests laufen** bei jedem Push und Pull Request
2. **Sicherheitsscans** prüfen den Code
3. **Versionsnummer** wird automatisch ermittelt (Semantic Versioning)
4. **PyPI Package** wird gebaut und hochgeladen
5. **GitHub Release** wird erstellt
6. **Dokumentation** wird deployed
7. **CHANGELOG** wird aktualisiert

## 📝 Commit Message Konvention

Verwende **Conventional Commits** für automatische Versionierung:

```bash
# Neue Features (erhöht Minor-Version: 1.2.0 → 1.3.0)
git commit -m "feat: neue Prognose-Funktion hinzugefügt"
git commit -m "feat(forecaster): unterstützung für neue Modelle"

# Bug Fixes (erhöht Patch-Version: 1.2.0 → 1.2.1)
git commit -m "fix: fehler in der Datenverarbeitung behoben"
git commit -m "fix(preprocessing): korrekte Behandlung von NaN-Werten"

# Breaking Changes (erhöht Major-Version: 1.2.0 → 2.0.0)
git commit -m "feat!: API komplett überarbeitet"
git commit -m "feat: neue Schnittstelle\n\nBREAKING CHANGE: alte API entfernt"

# Andere (erzeugen KEIN Release)
git commit -m "docs: README aktualisiert"
git commit -m "chore: Dependencies aktualisiert"
git commit -m "refactor: Code aufgeräumt"
git commit -m "test: weitere Tests hinzugefügt"
git commit -m "ci: Workflow verbessert"
```

### Commit-Typen

| Typ | Beschreibung | Release? |
|-----|--------------|----------|
| `feat:` | Neue Funktion | ✅ Minor |
| `fix:` | Bug Fix | ✅ Patch |
| `perf:` | Performance-Verbesserung | ✅ Patch |
| `refactor:` | Code-Umstrukturierung | ✅ Patch |
| `docs:` | Nur Dokumentation | ❌ Nein |
| `test:` | Tests hinzugefügt | ❌ Nein |
| `chore:` | Wartung, Dependencies | ❌ Nein |
| `ci:` | CI/CD Änderungen | ❌ Nein |
| `!` oder `BREAKING CHANGE:` | Breaking Change | ✅ Major |

## 🔧 Einrichtung (Einmalig)

### 1. GitHub Secrets einrichten

Gehe zu: `https://github.com/sequential-parameter-optimization/spotforecast2/settings/secrets/actions`

Erstelle ein Secret:
- **Name:** `PYPI_TOKEN`
- **Value:** Dein PyPI API Token (von https://pypi.org/manage/account/token/)

### 2. Branch Protection Rules (Empfohlen)

**Exakte Navigation (Schritt für Schritt):**

1. Gehe zu: https://github.com/sequential-parameter-optimization/spotforecast2
2. Klicke oben auf den **`Settings` Tab** (Zahnrad-Icon rechts oben)
3. Im **linken Menü** suchst du nach **`Branches`** (ist in der Sektion)
4. Klicke auf **`Add rule`** oder **`Add branch protection rule`**
5. Im **Feld "Branch name pattern"** schreibst du: `main`
6. Jetzt scrollst du nach unten und aktivierst diese Checkboxen:

**Minimale Sicherheit:**
- ✅ **Require a pull request before merging**
  - Wähle auch: ☑️ Require approvals: **1** (eine Person muss genehmigen)
- ✅ **Require status checks to pass before merging**
  - ➜ Warte bis unten neue Optionen erscheinen
  - ➜ Suche und wähle: `Test on Python 3.13` (MUSS grün sein!)
  - ➜ Optional auch: `Code Quality` und `Security Scan`
- ✅ **Require branches to be up to date before merging**
- ✅ **Require conversation resolution before merging** (optional)

7. Scrolle ganz nach unten rechts und klicke den grünen **`Create`** Button

**Fertig!** Branch ist jetzt geschützt. 🔒

> **💡 Detaillierte Anleitung mit visuellen Schritten:** Siehe [BRANCH_PROTECTION_GUIDE.md](BRANCH_PROTECTION_GUIDE.md)

**Für maximale Sicherheit (optional zusätzlich):**
- ✅ Require dismissals stale pull request approvals when new commits are pushed
- ✅ Lock branch (verhindert direkten Push)

### 3. GitHub Pages aktivieren

Gehe zu: `Settings → Pages`
- **Source:** Deploy from a branch
- **Branch:** `gh-pages` / `/ (root)`

## 👥 Tägliche Arbeit

> **💡 Bevorzugst du eine grafische Oberfläche?** Siehe [GITHUB_DESKTOP_GUIDE.md](GITHUB_DESKTOP_GUIDE.md) für die vollständige GitHub Desktop Anleitung!

### Arbeiten mit der Kommandozeile

### Feature entwickeln

```bash
# 1. Neuen Branch erstellen
git checkout -b feature/neue-funktion

# 2. Entwickeln und committen
git add .
git commit -m "feat: neue Prognose-Methode implementiert"

# 3. Push und Pull Request erstellen
git push origin feature/neue-funktion
```

### Pull Request erstellen

1. Gehe zu GitHub
2. Erstelle Pull Request von deinem Branch zu `main`
3. Warte auf grüne Tests ✅
4. Merge den Pull Request

### Release wird automatisch erstellt! 🎉

Nach dem Merge:
1. ⏱️ 2-3 Minuten warten
2. 🎁 Neues Release erscheint auf GitHub
3. 📦 Package ist auf PyPI verfügbar
4. 📚 Dokumentation ist aktualisiert

## 🔍 Monitoring

### Workflow-Status prüfen

- **Actions:** https://github.com/sequential-parameter-optimization/spotforecast2/actions
- **Releases:** https://github.com/sequential-parameter-optimization/spotforecast2/releases
- **PyPI:** https://pypi.org/project/spotforecast2/
- **Docs:** https://sequential-parameter-optimization.github.io/spotforecast2/

### Bei Fehlern

1. Gehe zu `Actions` Tab auf GitHub
2. Klicke auf den fehlgeschlagenen Workflow
3. Prüfe die Logs
4. Behebe das Problem in einem neuen Commit

## 🛡️ Sicherheit

### Automatische Sicherheitschecks

- **Dependabot:** Aktualisiert Dependencies automatisch wöchentlich
- **Safety:** Prüft auf bekannte Sicherheitslücken in Dependencies
- **Bandit:** Scannt Code auf Sicherheitsprobleme
- **CodeQL:** GitHub's Security Scanning (optional aktivierbar)

### Sicherheitsupdates

Dependabot erstellt automatisch Pull Requests für:
- Sicherheitsupdates (hohe Priorität)
- Dependency-Updates (wöchentlich)

Einfach die PRs prüfen und mergen.

## 📊 Versionierung

Semantic Versioning: `MAJOR.MINOR.PATCH`

**Beispiele:**
- `1.0.0` → `1.0.1` (Bug Fix)
- `1.0.1` → `1.1.0` (Neues Feature)
- `1.1.0` → `2.0.0` (Breaking Change)

## 🎓 Best Practices

### Commit-Messages

✅ **Gut:**
```bash
git commit -m "feat(forecaster): unterstützung für XGBoost-Modelle"
git commit -m "fix(data): korrekte Zeitzone-Konvertierung"
git commit -m "docs: API-Beispiele hinzugefügt"
```

❌ **Schlecht:**
```bash
git commit -m "updates"
git commit -m "fix bug"
git commit -m "wip"
```

### Pull Request Workflow

1. **Kleine, fokussierte PRs** - Einfacher zu reviewen
2. **Beschreibender Titel** - Erklärt die Änderung
3. **Tests hinzufügen** - Für neue Features
4. **Dokumentation aktualisieren** - Bei API-Änderungen

> **💡 Tipp:** Mit [GitHub Desktop](GITHUB_DESKTOP_GUIDE.md) kannst du Pull Requests direkt aus der Anwendung erstellen!

### Hotfix erstellen

Für dringende Bugfixes:

```bash
git checkout -b hotfix/kritischer-fehler main
git commit -m "fix: kritischer Sicherheitsfehler behoben"
git push origin hotfix/kritischer-fehler
# Pull Request erstellen und mergen
# → Automatisches Patch-Release (z.B. 1.2.3 → 1.2.4)
```

## 🔄 Entwicklungs-Branches (Optional)

Für größere Features:

```bash
# Develop-Branch verwenden
git checkout -b develop
git push origin develop

# Feature entwickeln
git checkout -b feature/grosse-aenderung develop
# ... entwickeln ...
git commit -m "feat: große neue Funktion (Teil 1)"

# Zu develop mergen
# Erst wenn alles fertig ist, develop → main mergen
```

## 📞 Support

### Häufige Fragen

**Q: Wie erstelle ich ein manuelles Release?**  
A: Nicht nötig! Jeder Merge zu `main` mit `feat:` oder `fix:` erstellt automatisch ein Release.

**Q: Wie überspringe ich ein Release?**  
A: Verwende Commit-Typen ohne Release: `docs:`, `chore:`, `test:`, `ci:`

**Q: Wie korrigiere ich eine falsche Version?**  
A: Git-Tag manuell löschen und neu pushen (selten nötig)

**Q: Tests schlagen fehl?**  
A: Prüfe den Actions-Tab, behebe Fehler lokal, pushe neuen Commit

**Q: PyPI-Upload schlägt fehl?**  
A: Prüfe ob `PYPI_TOKEN` Secret korrekt gesetzt ist

## 📝 Checkliste für neues Release

- [ ] Alle Tests grün ✅
- [ ] Pull Request reviewed (optional)
- [ ] CHANGELOG automatisch erstellt ✅
- [ ] Version automatisch erhöht ✅
- [ ] PyPI-Upload erfolgreich ✅
- [ ] GitHub Release erstellt ✅
- [ ] Dokumentation deployed ✅

Alles passiert **automatisch**! 🎉

## 🎯 Zusammenfassung

**Für Developer:**
1. Feature entwickeln
2. Conventional Commit verwenden
3. Pull Request erstellen
4. Mergen

**Der Rest passiert automatisch:**
- ✅ Tests
- ✅ Versionierung
- ✅ PyPI Release
- ✅ GitHub Release
- ✅ Dokumentation
- ✅ Security Scans

**Kosten:** 0 € (GitHub Actions ist kostenlos für Public Repos)

**Zeitaufwand:** 0 Minuten (nach Initial-Setup)

**Sicherheit:** ⭐⭐⭐⭐⭐ (Tests, Scans, automatische Updates)

## 🖥️ GitHub Desktop Nutzer

Bevorzugst du eine grafische Oberfläche statt Kommandozeile?

**Siehe [GITHUB_DESKTOP_GUIDE.md](GITHUB_DESKTOP_GUIDE.md)** für:
- 📸 Schritt-für-Schritt Anleitung mit visuellen Beschreibungen
- 🎯 Typische Workflows in GitHub Desktop
- 💡 Copy & Paste Commit-Message Templates
- ✅ Checklisten für jeden Workflow
- 🔧 Troubleshooting für häufige Probleme

Die Release-Strategie funktioniert **identisch** - egal ob du die Kommandozeile oder GitHub Desktop verwendest!
