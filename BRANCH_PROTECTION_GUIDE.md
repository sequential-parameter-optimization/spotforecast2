# Branch Protection Rules - Visuelle Anleitung

## 🎯 Schnelle Navigation

Branch Protection Rules sind unter **Settings → Branches** zu finden, nicht unter "Branch protection".

> **⚠️ WICHTIG:** Status Checks werden nur angezeigt, wenn die Workflows mindestens einmal gelaufen sind! Siehe [WORKFLOWS_SETUP.md](WORKFLOWS_SETUP.md) für detaillierte Anleitung.

## 📸 Schritt-für-Schritt mit Screenshots

### Schritt 1: Settings öffnen

```
GitHub Repo URL: https://github.com/sequential-parameter-optimization/spotforecast2
                                                                           ↓
Klicke oben auf "Settings" (Zahnrad-Icon rechts oben neben "Watch/Star")
```

### Schritt 2: Zum Branches Menü

Im **linken Sidebar Menü**:
```
Code (mit Symbol)
Issues
Pull requests
Actions          ← Das war früher hier
Deployments
Pages
Environments
Branches         ← HIER klicken! (Nicht "Code security and analysis")
Secrets and variables
Custom properties
Collaborators
...
```

### Schritt 3: Protection Rule erstellen

```
After clicking "Branches":
┌─────────────────────────────────────────────┐
│ Branch protection rules                      │
│ Setup a branch protection rule               │
│                                              │
│        [Add rule]  ← Klicke hier            │
└─────────────────────────────────────────────┘
```

### Schritt 4: Main Branch auswählen

```
Branch name pattern *
┌─────────────────────┐
│      main           │  ← Schreibe "main" hier
└─────────────────────┘

* Wildcard patterns allowed
```

### Schritt 5: Anforderungen aktivieren

Scrolle nach unten und aktiviere diese **Checkboxen**:

#### A) Require a pull request before merging ✅
```
☑️ Require a pull request before merging
   └─ ☑️ Require approvals
      └─ Number of approvals required: 1
   └─ ☑️ Require review from Code Owners
   └─ ☑️ Restrict who can push to matching branches
      └─ (Optional: nur bestimmte Nutzer)
```

#### B) Require status checks to pass ✅
```
☑️ Require status checks to pass before merging
   └─ ☑️ Require branches to be up to date before merging
   
   Status checks that must pass (Scrolle unten!)
   ┌────────────────────────────────────────────────┐
   │ ☑️ Test on Python 3.13  ← MUSS AKTIVIERT sein │
   │ ☑️ Test on Python 3.12                         │
   │ ☑️ Test on Python 3.11                         │
   │ ☑️ Test on Python 3.10                         │
   │ ☑️ Test on Python 3.9                          │
   │ ☑️ Code Quality                                │
   │ ☑️ Security Scan                               │
   └────────────────────────────────────────────────┘
```

#### C) Weitere (Optional)
```
☑️ Require conversation resolution before merging
☑️ Require code reviews before merging
☑️ Require status checks
☑️ Include administrators
☑️ Lock branch
☑️ Allow force pushes (NICHT empfohlen!)
☑️ Allow deletions
```

### Schritt 6: Speichern

```
Ganz nach unten scrollen
                    ┌────────────────┐
                    │    [Create]    │  ← Grüner Button
                    └────────────────┘
```

## ✅ Fertig!

Dein main Branch ist jetzt geschützt mit:
- ✅ Mindestens 1 Genehmigung nötig
- ✅ Alle Tests müssen grün sein (3.9-3.13)
- ✅ Branch muss aktuell sein
- ✅ Alle Gespräche müssen gelöst sein

## 🔍 Status Checks sind leer? Das ist normal! ⚠️

**Problem:** "No required checks - No checks have been added"

Das ist **kein Fehler**! Die Status Checks erscheinen NICHT automatisch. Sie müssen erst aktiviert werden:

### Schritt 1: Workflows müssen laufen
Die `.github/workflows/*.yml` Dateien müssen mindestens einmal ausgelöst werden.

1. Gehe zu deinem Repository: https://github.com/sequential-parameter-optimization/spotforecast2
2. Klicke oben auf den Tab **`Actions`**
3. Links siehst du die Workflows:
   - `CI Tests`
   - `Release`
   - `Documentation`
4. Wenn sie deaktiviert sind (grauer Status): Klicke `Enable on this repository`

### Schritt 2: Einen Test-Push machen
```bash
# Auf feature-branch wechseln (nicht main!)
git checkout -b test/actions-setup

# Leere Datei erstellen (oder Code ändern)
echo "# Test" >> test.md

# Committen und pushen
git add test.md
git commit -m "test: workflow setup"
git push origin test/actions-setup
```

### Schritt 3: Workflows laufen lassen
1. Gehe zu **Actions** Tab auf GitHub
2. Du siehst einen Running Workflow
3. Warte bis alle Workflows durchgelaufen sind (🟢 grün)
4. **Das dauert 5-10 Minuten!**

### Schritt 4: Jetzt gibt es Status Checks!

Gehe zurück zu `Settings → Branches → Add rule`:
```
Branch name pattern: main
✅ Require a pull request before merging
✅ Require status checks to pass before merging
   └─ Jetzt scrollst du nach unten und siehst:
   
   ☑️ Test on Python 3.13   ← Wähle diese!
   ☑️ Test on Python 3.12
   ☑️ Test on Python 3.11
   ... usw
```

### Schritt 5: Cleanup (optional)
```bash
# Test-Branch löschen (optional)
git checkout main
git branch -D test/actions-setup
git push origin --delete test/actions-setup
```

## 📋 Vollständige Reihenfolge für Setup

```
1. DEPLOY: .github/workflows/*.yml zu main pushen
   git add .github/
   git commit -m "ci: github actions workflows"
   git push origin main

2. TRIGGER: Einen Pull Request erstellen
   git checkout -b setup/initial
   git commit -m "docs: initial setup" --allow-empty
   git push origin setup/initial
   → Pull Request erstellen

3. WARTEN: Actions Tab überwachen
   - Workflow sollte automatic starten
   - ⏳ 5-10 Minuten warten
   - 🟢 Alle grün?

4. SCHÜTZEN: Branch Protection Rule erstellen
   Settings → Branches → Add rule
   - Branch: main
   - Status checks: Jetzt sichtbar!
   - Aktiviere: Test on Python 3.13
   - Speichere

5. MERGE: Den Setup PR mergen
   - Tests sind grün
   - Merge Pull Request
   - 🎉 Branch ist jetzt geschützt!
```

## 🚨 Wichtig: Richtige Reihenfolge!

❌ **FALSCH:**
1. Branch Protection Rule erstellen (ohne Workflows)
2. Status Checks hinzufügen (sind leer!)
3. Dann erst Workflows pushen

✅ **RICHTIG:**
1. Workflows zu GitHub pushen
2. Workflows mindestens einmal laufen lassen
3. Dann erst Branch Protection Rule mit Status Checks erstellen

## 🔗 Direkte Links für dein Repo

| Aktion | Link |
|--------|------|
| **Actions Workflows** | https://github.com/sequential-parameter-optimization/spotforecast2/actions |
| **Branch Protection** | https://github.com/sequential-parameter-optimization/spotforecast2/settings/branch_protection_rules |
| **Workflows Dateien** | https://github.com/sequential-parameter-optimization/spotforecast2/tree/main/.github/workflows |

## 💡 Schnelle Checkliste

- [ ] `.github/workflows/` Dateien existieren
- [ ] Workflows wurden gepusht (`git push`)
- [ ] Actions Tab zeigt mindestens einen durchgelaufenen Workflow
- [ ] Status Checks sind jetzt in der Liste sichtbar
- [ ] Branch Protection Rule erstellt mit Status Checks
- [ ] `main` Branch ist geschützt 🔒

## 🎓 Was die Rules machen

| Regel | Bedeutung |
|-------|-----------|
| Require a pull request | Niemand darf direkt auf main pushen |
| Require approvals | Mindestens eine Person muss genehmigen |
| Require status checks | Alle Tests müssen bestanden haben |
| Up to date | Branch muss aktuell mit main sein |
| Conversation resolution | Alle Kommentare müssen adressiert sein |

## 💡 Für 4-Personen Team empfohlen

**Minimale Konfiguration:**
```
✅ Require PR before merging
✅ Require 1 approval (der andere kümmert sich nicht darum)
✅ Require status checks: Test on Python 3.13
✅ Require branches to be up to date
```

**Maximale Sicherheit (aber mehr Overhead):**
```
✅ Require PR before merging
✅ Require 1-2 approvals
✅ Require status checks: Alle Python-Tests
✅ Require branches to be up to date
✅ Require conversation resolution
✅ Include administrators
```

## 🚀 Nächste Schritte

1. ✅ Branch Protection Rule erstellt
2. Committe deine Änderungen auf einen Feature-Branch
3. Erstelle einen Pull Request
4. Die Rules erzwingen automatisch die Anforderungen
5. Merge nur möglich wenn alles grün ist 🎉

## 📝 Video-Alternative

Falls du visueller lernst:
1. YouTube: "GitHub Branch Protection Rules Setup"
2. GitHub Docs: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule
