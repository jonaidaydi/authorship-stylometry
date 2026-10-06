# Sprechernotizen zur AP-Präsentation

Sprache: Deutsch. Folien 1–12 sind auf insgesamt 15 Minuten ausgelegt. Folie 13 ist ein Quellenanhang. Anschließend sind etwa 10 Minuten Diskussion vorgesehen. Die Zuordnung der Sprecher ist ein Vorschlag für die Probe.

## Folie 1: Autorschaftserkennung mit Stylometrie

**Jonaid Aydi und Kai S. Kurono · 1:00**

Wir untersuchen journalistische Artikel aus dem Reuters-Korpus. Unsere Frage ist, ob sich Texte von William Kazer, Tim Farrand und Patricia Commins anhand ihres Sprachgebrauchs unterscheiden lassen. Das Projekt verbindet eine Exploration mit Wortmerkmalen und einen ergänzenden Vergleich mit expliziten Stilmerkmalen. Wir zeigen zuerst die Daten und den Versuchsaufbau. Danach erklären wir die Entscheidungen der Modelle, vergleichen die Ergebnisse und betrachten konkrete Fehler. Der vollständige Projekttitel lautet: Automatische Autorschaftserkennung journalistischer Texte mittels stylometrischer Merkmale und klassischer Machine-Learning-Verfahren. Die Untersuchung gehört zum Kurs Advanced Python for NLP bei Rainer Osswald und Yulia Zinova. Alle Ergebnisse beziehen sich auf diese festgelegte Auswahl.

Quellen: https://github.com/jonaidaydi/authorship-stylometry

## Folie 2: Forschungsfrage

**Kai S. Kurono · 0:50**

Wortmerkmale und Stilmerkmale messen unterschiedliche Eigenschaften eines Textes. Wörter und Wortfolgen erfassen beispielsweise, welche Unternehmen oder Länder vorkommen. Stilmerkmale beschreiben unter anderem Satzlängen, Funktionswörter und Zeichensetzung. Beide können Autoren unterscheiden, aber Themen und Stil sind in einem journalistischen Korpus nicht sauber getrennt. Wir fragen deshalb, wie beide Repräsentationen auf denselben Dokumenten abschneiden. Es geht um eine Zuordnung zu drei bereits bekannten Autoren. Die PCA zeigt Ähnlichkeiten in zwei Dimensionen. Die eigentliche Vorhersage erfolgt im vollständigen Merkmalsraum. Diese Unterscheidung ist für die Interpretation der Abbildungen entscheidend.

Quellen: https://github.com/jonaidaydi/authorship-stylometry

## Folie 3: Daten und Trennung der Textmengen

**Kai S. Kurono · 1:30**

Reuter_50_50 enthält für jeden der fünfzig Autoren jeweils fünfzig Trainings- und fünfzig Testartikel. Wir verwenden drei Autoren, also insgesamt dreihundert Artikel. Je Autor halten wir die alphabetisch erste Trainingsdatei als Mystery-Text zurück. Die anderen neunundvierzig bilden die Referenz. So entsteht eine Referenzmenge von hundert siebenundvierzig Texten, eine Mystery-Menge von drei und ein offizieller Test von hundertfünfzig. Die Mystery-Texte werden für den größeren Test nicht wieder in die Referenz aufgenommen. Alle Verfahren haben damit dieselbe Informationsgrundlage. Auch Vokabular, Skalierung und PCA lernen ausschließlich aus der Referenzmenge. Die Dateinamen und Hashes im Repository machen diese Aufteilung nachvollziehbar. Ein Hashvergleich fand keine exakt identischen Texte zwischen Referenz und Auswertung. Ähnliche Meldungen sind damit aber nicht ausgeschlossen.

Quellen: https://doi.org/10.24432/C5DS42 und results/document_manifest.csv

## Folie 4: Wortrepräsentation mit TF-IDF

**Kai S. Kurono · 1:20**

Der Wortansatz beginnt mit einer eigenen Tokenisierung und dem englischen Snowball-Stemmer aus NLTK. Der Vektorisierer berücksichtigt einzelne Wörter sowie Folgen aus zwei oder drei Tokens. Von den tausend ausgewählten Merkmalen sind achthundertzweiunddreißig Unigramme, hundert einundfünfzig Bigramme und siebzehn Trigramme. TF-IDF gewichtet Merkmale anhand ihrer Häufigkeit im Dokument und ihrer Verbreitung über die Referenztexte. Danach normiert das Verfahren jeden Vektor auf Länge eins. Der nächste Nachbar ist der Referenzartikel mit dem kleinsten euklidischen Abstand. Dessen Autor wird vorhergesagt. Die häufigsten Merkmale der Autorenprofile enthalten China, Millionenbeträge und Unternehmensnamen. Das ist für die Zuordnung hilfreich, macht aber zugleich die Themenabhängigkeit sichtbar. Die ursprüngliche Kombination aus Stemming und englischer Stoppwortliste erzeugt einen dokumentierten Hinweis. Wir haben sie für diesen Vergleich beibehalten.

Quellen: https://www.nltk.org/api/nltk.stem.snowball.html und results/tfidf_author_profiles.csv

## Folie 5: Explizite Stilmerkmale

**Jonaid Aydi · 1:30**

Der ergänzende Ansatz verwendet sechsundzwanzig vorab festgelegte Merkmale. Fünf beschreiben die Textoberfläche, beispielsweise mittlere Satzlänge, Wortlänge und Type-Token-Ratio. Dazu kommen fünfzehn Funktionswörter und sechs Satzzeichen. Die Häufigkeiten beziehen sich auf tausend alphabetische Wörter. spaCy liefert die Tokenisierung und die Satzgrenzen. Da die Merkmale verschiedene Einheiten haben, standardisieren wir sie mit Mittelwert und Standardabweichung der Referenztexte. Der Test beeinflusst diese Werte nicht. Ein Merkmal, die Häufigkeit von Ausrufezeichen, ist in der Referenz konstant und entfällt. Übrig bleiben fünfundzwanzig Dimensionen. Relative Häufigkeiten lösen nicht jedes Längenproblem. Insbesondere die Type-Token-Ratio hängt weiterhin von der Textlänge ab. POS-N-Gramme gehören nicht zum untersuchten Merkmalsumfang.

Quellen: stylometry/features.py und results/standardisation.json

## Folie 6: Einzelner Nachbar oder Autorenzentrum

**Jonaid Aydi · 1:10**

Im ersten Stilverfahren bleibt die Entscheidungsregel dieselbe wie bei TF-IDF. Wir wählen den nächstgelegenen einzelnen Referenztext. Damit verändert sich vor allem die Repräsentation. Im zweiten Stilverfahren berechnen wir für jeden Autor den Mittelwert seiner standardisierten Referenzvektoren. Dieser Mittelwert ist das Autorenzentrum. Ein neuer Artikel bekommt das Label des nächstgelegenen Zentrums. Mittelwerte können Schwankungen einzelner Artikel ausgleichen. Sie können aber auch Besonderheiten eines Artikels verdecken. Beide Regeln bestehen aus überschaubaren NumPy-Operationen. Die PCA-Koordinaten gehen hier nicht ein. Auch beim Stilansatz berechnen wir die Abstände in allen fünfundzwanzig Dimensionen. Einen allgemeinen Vorteil von Autorenzentren können wir aus einem einzigen Vergleich noch nicht ableiten.

Quellen: stylometry/models.py und run_analysis.py

## Folie 7: PCA zeigt einen Ausschnitt der Varianz

**Kai S. Kurono · 1:30**

Die Abbildungen verwenden dieselben hundert siebenundvierzig Referenzartikel. Die drei Mystery-Texte sind durch größere Punkte markiert. Links sehen wir die TF-IDF-Repräsentation, rechts die Stilmerkmale. Die ersten beiden Komponenten zeigen zusammen rund elf Komma fünf acht Prozent der Wortvarianz und zwanzig Komma neun vier Prozent der Stilvarianz. Damit liegt der größte Teil der Variation außerhalb der Darstellung. Ein größerer dargestellter Anteil im Stilraum bedeutet auch nicht automatisch bessere Klassifikation. Die beiden Repräsentationen haben unterschiedliche Dimensionen und Varianzstrukturen. Wir verwenden die PCA, um Ähnlichkeiten und Überlappungen sichtbar zu machen. Die Vorhersagen werden weiterhin im vollständigen Raum berechnet. Die Reuters-HCA arbeitet ebenfalls auf vollständigen TF-IDF-Vektoren und erhält kondensierte euklidische Distanzen für Ward.

Quellen: results/tfidf_pca_coordinates.csv und results/style_pca_coordinates.csv

## Folie 8: Ergebnisse auf 150 offiziellen Testartikeln

**Jonaid Aydi · 1:10**

Die größere Auswertung ist die Grundlage unserer Leistungsaussage. TF-IDF ordnet hundert siebenunddreißig von hundertfünfzig Texten richtig zu. Das entspricht einundneunzig Komma drei drei Prozent. Der Stilansatz mit einzelnen Nachbarn erreicht hundertvier richtige Texte, also neunundsechzig Komma drei drei Prozent. Die Autorenzentren verbessern den Stilansatz auf hundertvierundzwanzig richtige Texte beziehungsweise zweiundachtzig Komma sechs sieben Prozent. Zum Vergleich: Weil jede Klasse fünfzig Artikel enthält, kommt eine konstante Vorhersage eines Autors auf dreiunddreißig Komma drei drei Prozent. Alle drei Verfahren liegen darüber. Die Macro-F1-Werte berücksichtigen die drei Autoren gleichgewichtet und bestätigen die Rangfolge. Wir berichten alle vorher festgelegten Modelle. Wir haben keine Parameter anhand dieser Testergebnisse optimiert.

Quellen: results/model_comparison.csv

## Folie 9: Verwechslungen beim Stil-Autorenzentrum

**Jonaid Aydi · 1:20**

Die Konfusionsmatrix zeigt, welche Autoren verwechselt werden. Die Zeilen enthalten die wahren Autoren und die Spalten die Vorhersagen. Die Diagonale zählt richtige Zuordnungen. Von Kazers fünfzig Artikeln werden vierzig richtig erkannt, acht Farrand und zwei Commins zugeordnet. Bei Farrand sind neununddreißig richtig und elf gehen an Kazer. Bei Commins sind fünfundvierzig richtig. Die Recall-Werte betragen damit achtzig, achtundsiebzig und neunzig Prozent. Die meisten Verwechslungen liegen zwischen Kazer und Farrand. Der TF-IDF-Ansatz hat ein anderes Fehlermuster: Zehn seiner dreizehn Fehler ordnen Commins-Artikel Farrand zu. Eine einzige Gesamtquote verdeckt solche Unterschiede. Als Nächstes schauen wir deshalb auf konkrete Dokumente.

Quellen: results/official_test_style_centroid_confusion.csv und results/official_test_tfidf_1nn_confusion.csv

## Folie 10: Drei konkrete Fehlerfälle

**Jonaid Aydi · 1:20**

Die Fälle stammen alle von Patricia Commins. Wir haben innerhalb dreier Ergebniskategorien jeweils den ersten sortierten Dateipfad gewählt. Das ist eine nachträgliche Illustration und keine repräsentative Stichprobe. Im ersten Fall berichtet Commins über Ladenschließungen bei Dayton Hudson. TF-IDF findet einen Farrand-Artikel über Burton, weil Wörter wie store und retail beiden Texten gemeinsam sind. Beide Stilmodelle liegen richtig. Im zweiten Fall geht es um Quaker und Snapple. Ein thematisch eng verwandter Commins-Artikel macht die lexikalische Zuordnung richtig. Das Stilzentrum entscheidet sich dagegen knapp für Kazer. Im dritten Fall geht es um Maisverarbeitung. Alle Modelle sagen Farrand voraus. Ein seltenes Satzzeichenmerkmal trägt besonders stark zum falschen Stilentscheid bei. Diese Fälle zeigen, dass Themenähnlichkeit sowohl richtige als auch falsche Wortentscheidungen unterstützen kann.

Quellen: results/qualitative_cases.json und results/README.md

## Folie 11: Python-Umsetzung und Reproduktion

**Kai S. Kurono · 1:10**

Die Umsetzung bleibt bewusst überschaubar. prepare_data lädt und filtert den Korpus. run_analysis führt den vollständigen Reuters-Vergleich aus. Kleine Module trennen das Laden der Texte, die Merkmalsextraktion und die Distanzberechnung. CSV-Dateien enthalten die Merkmale und Vorhersagen. JSON-Dateien dokumentieren Parameter, Versionen und Hashes. Damit lassen sich Bericht und Berechnung zusammen prüfen. Der Kursbezug liegt unter anderem in Funktionen mit Docstrings, Dictionaries, virtuellen Umgebungen, NLTK, spaCy, pandas und NumPy. Die neue Installation reproduziert die früheren Vorhersagen vollständig. Fünf gezielte Funktionstests und alle drei Notebooks laufen durch. Die Federalist-Beispiele nutzen jetzt denselben bereinigten Textbestand. Ihre Kategorien unterscheiden eindeutige, umstrittene und gemeinsame Autorschaft. Sie dienen als historisches Zusatzbeispiel.

Quellen: README_de.md und REPRODUCIBILITY.md

## Folie 12: Ergebnis, Grenzen und Beiträge

**Jonaid Aydi und Kai S. Kurono · 1:10**

Unsere Untersuchung zeigt, dass beide Repräsentationen für diese drei Reuters-Autoren nützliche Informationen enthalten. TF-IDF erzielt das beste Testergebnis. Autorenzentren verbessern den Stilansatz gegenüber einzelnen Nachbarn. Daraus folgt keine allgemeine Erkennung eines reinen persönlichen Stils. Themenunterschiede, verwandte Meldungen, redaktionelle Vorgaben und Texteigenschaften bleiben mögliche Einflüsse. Eine sinnvolle Fortsetzung wäre eine neue Autorenauswahl oder eine Auswertung mit kontrollierten Themen. Kai verantwortet die Wortrepräsentationen und die Exploration mit PCA, Clustering und Mystery-Zuordnung. Jonaid verantwortet den ergänzenden Stilvergleich, die Testauswertung und die Integration für reproduzierbare Läufe. Codex unterstützte die Integration, Prüfung und Vorbereitung der Unterlagen. Wir beide verantworten die Abgabe und ihre Erklärung. Damit öffnen wir die Diskussion.

Quellen: https://github.com/jonaidaydi/authorship-stylometry

## Folie 13: Quellen und weitere Details

**Beide · Anhang**

Diese Folie dient als Anhang für die Diskussion. Die Datenquelle ist Reuter_50_50 von UCI. Die wichtigsten Implementierungsquellen sind die offiziellen Dokumentationen der verwendeten Bibliotheken. Das Repository enthält die Berichte, die vollständigen Ergebnistabellen und die genaue Datenaufteilung. Die Folien 1 bis 12 sind für insgesamt etwa fünfzehn Minuten vorgesehen. Anschließend sind ungefähr zehn Minuten Diskussion eingeplant.

Quellen: https://github.com/jonaidaydi/authorship-stylometry

## Fragen für die Diskussion

**Warum nur drei Autoren?**

Die feste Auswahl folgt dem kompakten Projektumfang. Die Resultate sind keine Aussage über alle 50 Autoren.

**Weshalb bleibt die Referenz bei 147 Artikeln?**

So vergleichen alle Verfahren dieselben Referenztexte. Die drei Mystery-Artikel werden für den offiziellen Test nicht wieder aufgenommen.

**Warum skaliert ihr die Stilmerkmale?**

Satzlänge, Wortlänge und relative Häufigkeiten haben unterschiedliche Einheiten. Referenzmittelwert und Referenzstandardabweichung machen die Abstände vergleichbarer.

**Ist TF-IDF überwacht?**

Die Vektorisierung lernt keine Autorenlabels. Die anschließende Zuordnung nutzt aber die bekannten Labels der Referenzartikel.

**Warum ist PCA keine Leistungsbewertung?**

Zwei Komponenten zeigen nur einen Teil der Varianz. Sichtbare Gruppen beweisen weder richtige Testvorhersagen noch Themenunabhängigkeit.

**Was heißt Macro-F1?**

Man berechnet F1 für jeden Autor und mittelt diese drei Werte gleichgewichtet. Precision und Recall gehen beide in F1 ein.

**Sind die Texte frei von Leakage?**

Die gelernten Transformationen verwenden nur Referenztexte, und exakte Referenz/Test-Duplikate wurden ausgeschlossen. Ähnliche Meldungen und Themenüberschneidungen bleiben möglich.

**Welche Merkmale sind am wichtigsten?**

Für konkrete Zentroidentscheidungen lässt sich der Beitrag jedes Merkmals zur quadrierten Distanzdifferenz berechnen. Daraus folgt keine globale kausale Bedeutung. Eine systematische Ablation wurde nicht durchgeführt.

**Warum keine POS-N-Gramme?**

Der feste Vergleich untersucht die dokumentierten 26 Stilmaße. Eine weitere Merkmalsgruppe wäre ein neues Experiment und sollte nicht anhand dieses bereits betrachteten Tests optimiert werden.

**Was wurde an Federalist korrigiert?**

Beide Notebooks nutzen denselben Parser, getrennte Metadaten und Kategorien. Von Essay 70 bleibt die erste Fassung. Ward erhält kondensierte Distanzen.

**Was muss noch persönlich erledigt werden?**

Die Abgabe und Erklärung verantworten beide Mitwirkenden. Den Vortrag gemeinsam proben und Bericht sowie Repository gemäß Kursvorgaben einreichen. GitHub-Veröffentlichung ersetzt die persönliche Abgabe nicht.
