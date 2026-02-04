# GitHub Desktop Integration für spotforecast2

## 🖥️ GitHub Desktop Setup

GitHub Desktop ist eine benutzerfreundliche GUI-Alternative zur Kommandozeile und funktioniert perfekt mit unserem automatisierten Release-Workflow.

## 📥 Installation

1. Download: https://desktop.github.com/
2. Installieren und GitHub-Account verbinden
3. Repository klonen: `File → Clone Repository → GitHub.com → spotforecast2`

## 🎯 Tägliche Arbeit mit GitHub Desktop

### 1. Feature-Branch erstellen

**Schritt-für-Schritt:**

1. **Stelle sicher, dass du auf `main` bist**
   - Oben: Aktueller Branch sollte "main" zeigen
   - Falls nicht: `Current Branch → main`

2. **Branch aktualisieren**
   - Klicke `Fetch origin` (oben rechts)
   - Falls Updates verfügbar: Klicke `Pull origin`

3. **Neuen Branch erstellen**
   - Klicke `Current Branch` (oben)
   - Klicke `New Branch`
   - Name eingeben: z.B. `feature/neue-prognose-methode`
   - "Create branch based on: main"
   - Klicke `Create Branch`

### 2. Änderungen committen

**Schritt-für-Schritt:**

1. **Code ändern** in VS Code oder deinem Editor

2. **Änderungen ansehen**
   - GitHub Desktop zeigt alle geänderten Dateien links
   - Klicke auf eine Datei um Diff anzusehen
   - Grün = hinzugefügt, Rot = gelöscht

3. **Commit-Message schreiben**
   
   **WICHTIG:** Verwende Conventional Commits Format!
   
   Links unten im "Summary" Feld:
   ```
   feat: neue Prognose-Methode für XGBoost
   ```
   
   Im "Description" Feld (optional):
   ```
   - Unterstützung für XGBoost-Modelle
   - Automatische Hyperparameter-Optimierung
   - Tests hinzugefügt
   ```

4. **Commit erstellen**
   - Klicke den blauen Button `Commit to feature/neue-prognose-methode`

### 3. Änderungen pushen

**Schritt-für-Schritt:**

1. **Nach dem Commit**
   - Oben rechts erscheint: `Push origin` oder `Publish branch`
   
2. **Pushen**
   - Klicke `Push origin` (oder `Publish branch` beim ersten Mal)
   - GitHub Desktop lädt deine Änderungen hoch

### 4. Pull Request erstellen

**Schritt-für-Schritt:**

1. **Nach dem Push**
   - GitHub Desktop zeigt: "Create Pull Request"
   - Klicke `Create Pull Request`
   - Browser öffnet sich mit GitHub

2. **Pull Request ausfüllen**
   - **Title:** Wird automatisch vom ersten Commit übernommen
   - **Description:** Beschreibe die Änderungen
   - **Base:** sollte `main` sein
   - **Compare:** dein Feature-Branch

3. **Pull Request erstellen**
   - Klicke `Create pull request`
   - ✅ Tests laufen automatisch!

4. **Warten auf grüne Checks**
   - Alle Tests müssen grün sein ✅
   - Bei Fehlern: Siehe Logs, behebe Fehler, pushe neuen Commit

5. **Merge**
   - Klicke `Merge pull request`
   - Klicke `Confirm merge`
   - 🎉 **Automatisches Release wird erstellt!**

## 📝 Commit-Message Templates für GitHub Desktop

### Copy & Paste Vorlagen

**Neue Funktion (Minor-Version):**
```
feat: [Kurze Beschreibung]
```

**Mit Modul:**
```
feat(forecaster): [Beschreibung]
```

**Bug Fix (Patch-Version):**
```
fix: [Kurze Beschreibung]
```

**Breaking Change (Major-Version):**
```
feat!: [Kurze Beschreibung]
```

**Dokumentation (kein Release):**
```
docs: [Kurze Beschreibung]
```

**Tests (kein Release):**
```
test: [Kurze Beschreibung]
```

**Refactoring (Patch-Version):**
```
refactor: [Kurze Beschreibung]
```

### Beispiele für Summary-Feld

✅ **Gut:**
```
feat(forecaster): XGBoost-Modell-Unterstützung
fix(preprocessing): NaN-Werte korrekt behandelt
docs: API-Dokumentation erweitert
test: Tests für Daten-Import hinzugefügt
refactor(utils): Code vereinfacht
```

❌ **Schlecht:**
```
Update
Fixed bug
WIP
Änderungen
test
```

## 🔄 Typische Workflows

### Workflow 1: Feature hinzufügen

1. `Current Branch → New Branch`
2. Name: `feature/beschreibung`
3. Code schreiben
4. Änderungen ansehen in GitHub Desktop
5. Summary: `feat: neue Funktionalität`
6. `Commit to feature/beschreibung`
7. `Push origin`
8. `Create Pull Request`
9. Auf GitHub mergen
10. ✅ Automatisches Release (z.B. 1.2.0 → 1.3.0)

### Workflow 2: Bug fixen

1. `Current Branch → New Branch`
2. Name: `fix/bug-beschreibung`
3. Bug fixen
4. Summary: `fix: [Problem] behoben`
5. `Commit to fix/bug-beschreibung`
6. `Push origin`
7. `Create Pull Request`
8. Auf GitHub mergen
9. ✅ Automatisches Release (z.B. 1.2.0 → 1.2.1)

### Workflow 3: Dokumentation aktualisieren

1. `Current Branch → New Branch`
2. Name: `docs/update`
3. Dokumentation schreiben
4. Summary: `docs: [was aktualisiert]`
5. `Commit to docs/update`
6. `Push origin`
7. `Create Pull Request`
8. Auf GitHub mergen
9. ℹ️ **Kein Release**, nur Doku-Update

### Workflow 4: Mehrere Commits im gleichen Branch

1. Erster Commit:
   - Summary: `feat: Teil 1 implementiert`
   - `Commit to feature/xyz`
   - `Push origin`

2. Zweiter Commit:
   - Mehr Code schreiben
   - Summary: `feat: Teil 2 implementiert`
   - `Commit to feature/xyz`
   - `Push origin`

3. Dritter Commit:
   - Summary: `test: Tests für neue Funktion`
   - `Commit to feature/xyz`
   - `Push origin`

4. Pull Request erstellen (alle Commits sind drin)
5. Mergen → Release basiert auf allen `feat:` und `fix:` Commits

## 🔧 Nützliche GitHub Desktop Features

### History ansehen

- `History` Tab (links)
- Zeigt alle Commits
- Klicke einen Commit um Details zu sehen

### Änderungen verwerfen

- Rechtsklick auf Datei
- `Discard changes` (Vorsicht: unwiderruflich!)

### Stash (Änderungen temporär speichern)

- `Branch → Stash all changes`
- Später: `Branch → Restore stashed changes`

### Branch wechseln

- `Current Branch` (oben)
- Branch auswählen
- GitHub Desktop warnt bei uncommitted changes

### Updates holen

- `Fetch origin` (regelmäßig klicken!)
- Zeigt ob neue Commits verfügbar sind
- `Pull origin` um Updates zu holen

### Branch löschen

- Nach dem Merge: GitHub Desktop bietet an, Branch zu löschen
- Oder: `Branch → Delete` (nur lokale Branches)

## 🎨 Visual Studio Code Integration

GitHub Desktop arbeitet perfekt mit VS Code zusammen:

### In VS Code öffnen

- `Repository → Open in Visual Studio Code`
- Oder: Keyboard Shortcut `Cmd+Shift+A` (Mac) / `Ctrl+Shift+A` (Windows)

### VS Code Terminal

Du kannst trotzdem die Kommandozeile verwenden:
```bash
# In VS Code Terminal
git status
git log
pytest tests/
```

## ⚠️ Wichtige Hinweise

### Conventional Commits sind PFLICHT

Für automatische Releases **muss** die Commit-Message das richtige Format haben:

- `feat:` → Neues Feature → Minor-Version
- `fix:` → Bug Fix → Patch-Version
- `feat!:` → Breaking Change → Major-Version
- `docs:`, `test:`, `chore:` → Kein Release

### Immer auf aktuellem Stand bleiben

Vor jedem neuen Branch:
1. `Current Branch → main`
2. `Fetch origin`
3. `Pull origin`
4. Dann neuen Branch erstellen

### Pull Requests nicht lokal mergen

**Nicht** in GitHub Desktop mergen, sondern:
- Immer auf **GitHub.com** mergen
- Damit die Workflows laufen!

### Nach dem Merge

1. `Current Branch → main`
2. `Pull origin` (holt das Merge-Commit)
3. Alten Feature-Branch löschen (GitHub Desktop fragt automatisch)

## 🐛 Troubleshooting

### "Push rejected" Fehler

**Problem:** Jemand anders hat inzwischen gepusht

**Lösung:**
1. `Repository → Pull`
2. Falls Konflikte: In VS Code lösen
3. Nochmal pushen

### Commit-Message vergessen

**Problem:** Falsches Format verwendet

**Lösung:** Vor dem Push:
1. `History` Tab
2. Rechtsklick auf letzten Commit
3. `Amend commit` (ändert letzte Message)
4. Neue Message: `feat: korrekte Beschreibung`

### Zu viele Änderungen

**Problem:** Viele Dateien gleichzeitig geändert

**Lösung:**
- Kannst Checkboxen bei Dateien deaktivieren
- Nur ausgewählte Dateien werden committed
- Rest bleibt für nächsten Commit

### Branch ist behind main

**Problem:** Main hat neue Commits

**Lösung:**
1. `Branch → Update from main`
2. GitHub Desktop mergt automatisch
3. Falls Konflikte: In VS Code lösen

## 📋 Checkliste für jeden Workflow

**Vor dem Start:**
- [ ] Auf `main` Branch
- [ ] `Fetch origin` geklickt
- [ ] `Pull origin` (falls Updates)
- [ ] Neuen Feature-Branch erstellt

**Während der Arbeit:**
- [ ] Änderungen regelmäßig committen
- [ ] Conventional Commit Format verwendet
- [ ] Beschreibende Commit-Messages

**Vor dem Push:**
- [ ] Änderungen reviewed in GitHub Desktop
- [ ] Commit-Messages nochmal prüfen
- [ ] Alle Tests lokal ausgeführt (optional)

**Pull Request:**
- [ ] Push durchgeführt
- [ ] PR auf GitHub erstellt
- [ ] Beschreibung ausgefüllt
- [ ] Warten auf grüne Checks ✅
- [ ] Merge auf GitHub.com (nicht in Desktop!)

**Nach dem Merge:**
- [ ] Zurück zu `main` in GitHub Desktop
- [ ] `Pull origin` um Updates zu holen
- [ ] Feature-Branch löschen
- [ ] 2-3 Minuten warten → Neues Release ist live! 🎉

## 🎓 Best Practices

### Klein committen, oft pushen

```
✅ Gut:
Commit 1: feat: Basis-Implementierung
Commit 2: test: Tests hinzugefügt
Commit 3: docs: Dokumentation erweitert
Commit 4: refactor: Code vereinfacht

❌ Schlecht:
Commit 1: Alles fertig (500 Zeilen geändert)
```

### Beschreibende Branch-Namen

```
✅ Gut:
feature/xgboost-support
fix/nan-handling-preprocessing
docs/api-examples

❌ Schlecht:
test
fix
my-branch
branch-123
```

### Regelmäßig updaten

- Jeden Morgen: `Fetch origin` + `Pull origin` auf main
- Vor neuem Branch: Immer aktuellen Stand holen
- Während der Arbeit: Gelegentlich `Fetch` klicken

## 📺 Visueller Workflow

```
GitHub Desktop                  Browser (GitHub.com)
─────────────────               ───────────────────

1. Current Branch → main
   Pull origin
                                
2. New Branch
   "feature/neue-funktion"
   
3. [Code schreiben in VS Code]

4. Änderungen sichtbar
   ✓ File1.py
   ✓ File2.py
   
5. Summary:
   "feat: neue Funktion"
   
6. Commit Button
   
7. Push origin →              
                                8. Create PR Button
                                   
                                9. Fill in details
                                   
                                10. Create PR
                                    
                                11. ✅ Tests laufen
                                    
                                12. Merge Button
                                    
                                13. 🎉 Release!

14. Switch to main
    Pull origin
    
15. ✓ Neues Release lokal
```

## 🔗 Wichtige Links

- **GitHub Desktop Docs:** https://docs.github.com/en/desktop
- **Unser Repository:** https://github.com/sequential-parameter-optimization/spotforecast2
- **Release-Strategie:** Siehe RELEASE_MANAGEMENT.md
- **Quick Guide:** Siehe .github/WORKFLOWS_GUIDE.md

## 💡 Tipps für Anfänger

1. **Keine Angst vor Fehlern** - Branches sind sicher zum experimentieren
2. **Lieber zu oft committen** als zu selten
3. **Beschreibende Messages** helfen dem ganzen Team
4. **Bei Unsicherheit** → Frag im Team nach
5. **Tests lokal laufen lassen** bevor du pushst (optional aber gut)

## 🎯 Zusammenfassung

**GitHub Desktop macht es einfach:**
1. ✅ Visuell - Siehst alle Änderungen
2. ✅ Intuitiv - Kein Kommandozeilen-Wissen nötig
3. ✅ Sicher - Schwer etwas kaputt zu machen
4. ✅ Funktioniert perfekt mit unserem automatischen Release-Workflow

**Der Release-Prozess bleibt gleich:**
- Conventional Commits verwenden
- Pull Request auf GitHub mergen
- Automatisches Release passiert! 🚀

**Viel Erfolg! 🎉**
