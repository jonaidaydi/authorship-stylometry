# Untersuchung von Autorschaft mit PCA und hierarchischer Clusteranalyse

**Deutsch** | [English](README.md)

Kursprojekt von **Jonaid Aydi und Kai S. Kurono**, *Advanced Python for NLP* (2026),
Rainer Osswald und Yulia Zinova.

Das Projekt untersucht, wie sich Texte verschiedener Autoren in wortbasierten
Darstellungen verteilen. Hauptkomponentenanalyse (HKA, englisch PCA) und
hierarchische Clusteranalyse (HCA) machen Ähnlichkeiten und Überschneidungen
sichtbar. Die Federalist Papers und Reuters-Artikel bilden zwei Fallbeispiele.

## Notebooks

| Notebook | Untersuchung |
|---|---|
| [HKA Federalist Papiere.ipynb](HKA%20Federalist%20Papiere.ipynb) | TF-IDF, zweidimensionale PCA, Aufsatznummern und Wortladungen |
| [HCA Federalist Papiere.ipynb](HCA%20Federalist%20Papiere.ipynb) | Hierarchische Clusteranalyse von Worthäufigkeitsvektoren mit deaktivierter IDF-Gewichtung |
| [reuter HKA, HCA.ipynb](reuter%20HKA%2C%20HCA.ipynb) | Stemming, TF-IDF mit Wort-n-Grammen, PCA, hierarchische Clusteranalyse und Zuordnung zum nächstgelegenen Text |

Das Reuters-Notebook wählt derzeit William Kazer, Tim Farrand und
Patricia Commins aus. Pro Autor wird die erste Datei in sortierter Reihenfolge
als Mystery-Text zurückgehalten. Die übrigen 49 Artikel bilden die Referenztexte.
Vektorisierer und PCA werden an diesen 147 Referenzartikeln angepasst.
Die drei zurückgehaltenen Texte werden anschließend mit den angepassten
Modellen transformiert. Die Zuordnung erfolgt über den nächstgelegenen
Referenztext im vollständigen TF-IDF-Raum.

## Installation und Start

Python 3 verwenden und die folgenden Befehle im Hauptordner des Repositorys
ausführen. Installation und erster Korpusdownload benötigen Internetzugang.

Unter Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyterlab
```

Unter macOS oder Linux für die letzten beiden Befehle `.venv/bin/python` verwenden.

Ein Notebook in JupyterLab öffnen und alle Zellen mit einem frischen Kernel
von oben nach unten ausführen. Die Notebooks im Hauptordner belassen, weil
sich ihre Datenpfade auf diesen Ordner beziehen. Das Reuters-Notebook enthält
eine zusätzliche Installationszelle für seaborn, das bereits in
`requirements.txt` aufgeführt ist.

Die Abhängigkeitsliste folgt den aktuellen Imports. Eine neue Installation
mit vollständiger Ausführung aller drei Notebooks wurde für diesen
Repositorystand noch nicht geprüft. Die Paketversionen sind noch nicht festgelegt.

## Daten vorbereiten

### Federalist Papers

Jedes Federalist-Notebook lädt in seiner ersten Codezelle den
[Text von Project Gutenberg](https://www.gutenberg.org/cache/epub/18/pg18.txt)
als `federalist.txt` herunter. Das PCA-Notebook schreibt zusätzlich einzelne
Aufsätze nach `federalist_papers/`. Beide Ablageorte werden von Git ignoriert.

### Reuters

[Reuter_50_50 bei UCI](https://archive.ics.uci.edu/dataset/217/reuter%2B50%2B50)
über die Schaltfläche Download herunterladen oder direkt das
[ZIP-Archiv](https://archive.ics.uci.edu/static/public/217/reuter%2B50%2B50.zip)
verwenden. Den Inhalt neben den Notebooks in einen Ordner mit dem genauen
Namen `reuter+50+50` entpacken.

Danach müssen unter anderem diese Pfade vorhanden sein:

```text
reuter+50+50/
  C50train/
    WilliamKazer/
    TimFarrand/
    PatriciaCommins/
    ... weitere Autoren
  C50test/
    ... Autoren
```

Das aktuelle Reuters-Notebook liest nur `C50train`. Seine drei Mystery-Texte
werden aus diesem Teil zurückgehalten. Der offizielle Testteil `C50test`
wird nicht ausgewertet. Korpus und ZIP-Archiv bleiben lokal.

## Abbildungen und Ergebnisse lesen

Die Federalist-Notebooks zeigen PCA-Diagramme und ein Dendrogramm.
Das Reuters-Notebook speichert zusätzlich `hca_dendrogram.png` und
`hka_pca_biplot.png`. Danach gibt es seine drei Zuordnungen und die
höchstgewichteten Merkmale pro Autor aus. Erzeugte Bilder werden von Git ignoriert.

Die PCA zeigt nur zwei Dimensionen. An den Achsen steht der Anteil der
dargestellten Varianz. Sichtbare Trennung kann sowohl mit Themenwörtern als
auch mit Unterschieden zwischen Autoren zusammenhängen. Drei Mystery-Texte
sind Anschauungsbeispiele und belegen keine allgemeine Klassifikationsgenauigkeit.

## Aktuelle methodische Prüfpunkte

Die Notebooks bewahren den aktuellen Erkundungsstand. Vor einer abschließenden
Auswertung müssen folgende Punkte geklärt werden:

- Die Federalist-Merkmalsvektoren enthalten noch Überschriften und Autorenangaben.
  Diese müssen von den eigentlichen Aufsatztexten getrennt werden.
- Die beiden Federalist-Notebooks verwenden unterschiedliche Regeln für
  umstrittene Autorschaft. Ihre Zuordnungen und der Umgang mit dem doppelten
  Aufsatz 70 müssen vereinheitlicht und dokumentiert werden.
- Das Federalist-HCA-Notebook übergibt eine quadratische Distanzmatrix an
  Ward-Linkage. SciPy interpretiert deren Zeilen als Beobachtungen. Für das
  Clustering der ursprünglichen Wortvektoren werden die Merkmalsmatrix oder
  komprimierte euklidische Distanzen benötigt.
  Siehe die [SciPy-Dokumentation zu linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

## Beiträge

| Mitwirkende | Beitrag |
|---|---|
| Kai S. Kurono | Federalist- und Reuters-Notebooks, Wortrepräsentationen, PCA, hierarchische Clusteranalyse und Zuordnung der Reuters-Mystery-Texte. |
| Jonaid Aydi | Repositoryorganisation, Abhängigkeitsliste, Installationsanleitung sowie englische und deutsche Dokumentation. |

## Datenquellen

- Hamilton, Madison und Jay. *The Federalist Papers*.
  [Project Gutenberg, eBook 18](https://www.gutenberg.org/ebooks/18).
- Liu, Z. (2006). *Reuter_50_50*. UCI Machine Learning Repository.
  [DOI: 10.24432/C5DS42](https://doi.org/10.24432/C5DS42).
  Die UCI-Datensatzseite nennt CC BY 4.0.
