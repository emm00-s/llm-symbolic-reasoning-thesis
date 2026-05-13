# Revisione repository: logica di sperimentazione e metodologia

## Sintesi
La pipeline è coerente con un obiettivo di **benchmark diagnostico controllato** per invarianti logiche sotto variazione narrativa. I punti metodologici forti sono:

- uso di template formali con label gold verificate via Z3;
- controbilanciamento sistematico della posizione delle opzioni A/B/C/D;
- separazione tra metrica sampled (comportamento stocastico) e argmax (comportamento deterministico);
- analisi multi-livello (run, puzzle, template, variante) con bootstrap a cluster.

## Verifica della logica sperimentale

### 1) Disegno del dataset
- 9 template × 4 varianti = 36 puzzle, con varianti narrative (`familiar`, `belief_violating`, `artificial`, `abstract`).
- Questo è adeguato a testare **invarianza intra-template**.
- La dimensione rimane piccola: utile per diagnosi, non per stime robuste di generalizzazione ampia.

### 2) Randomizzazione e controbilanciamento
- `run.py` ordina stabilmente i puzzle per `id` e usa `option_order_for(...)` per assegnare la permutazione delle label alle lettere.
- Con 36 puzzle e 10 seed, si ottiene copertura uniforme delle 24 permutazioni (come documentato nel README).
- Questa scelta riduce in modo forte il confondimento dovuto a bias posizionale.

### 3) Definizione delle misure di outcome
- `correct_sampled`: confronta la label campionata con la gold; misura comportamento realizzato in sampling.
- `correct_argmax`: confronta la label più probabile con la gold; misura qualità della distribuzione senza rumore di campionamento.
- `is_invalid`: controllo di coerenza parser/wrapper; utile come sanity check operativo.

### 4) Statistica di analisi
- `analyze.py` applica bootstrap a cluster per `puzzle_id`, appropriato perché le unità ripetute (seed) non sono indipendenti.
- Calcola anche majority-vote e stabilità per puzzle, più consistency tra varianti.
- Buona distinzione fra diagnostica di confidenza (`max_prob`, `entropy`) e accuratezza.

## Punti di forza metodologici
- **Controllo sperimentale elevato** su confondenti di formato (ordine opzioni).
- **Tracciamento completo** delle variabili per audit (letter-level e label-level).
- **Repliche per seed** che permettono misure di stabilità.
- **Analisi gerarchica** coerente con la struttura del dataset.

## Limiti e rischi interpretativi
1. **N ridotto (36 puzzle)**: intervalli e confronti tra varianti possono essere instabili.
2. **Dipendenza dal parser a lettera**: anche con vincolo forte, il parsing a singolo token può introdurre errori residui in edge-case.
3. **Confronti multipli**: molte breakdown (template, category, variant) aumentano rischio di over-interpretazione esplorativa.
4. **Tie handling nel majority vote**: tie-break deterministico evita non-determinismo, ma può introdurre lieve bias sistematico.
5. **Modello singolo / configurazione singola**: inferenza limitata se non si replica su più modelli e hyperparameter grid.

## Raccomandazioni pratiche (priorità)
1. **Aggiungere split di replicazione**: ripetere run completi con diversi `temperature/top_p` e almeno 2 modelli addizionali.
2. **Esplicitare protocollo inferenziale**: predefinire metriche primarie (es. sampled accuracy + cross-variant consistency) e secondarie.
3. **Gestire multiple comparisons**: riportare almeno FDR o dichiarare esplicitamente natura esplorativa delle tabelle secondarie.
4. **Rafforzare validazione parser**: test unitari su output rumorosi (es. `"A."`, `"Option B"`, testo multilinea).
5. **Report standardizzato**: aggiungere script che produce automaticamente un report markdown con headline metric + CI.

## Valutazione complessiva
Metodologia **solida per un benchmark diagnostico small-scale** e ben allineata alla domanda di ricerca su invarianza logica sotto variazione narrativa. Le principali aree di miglioramento riguardano potenza statistica, controllo dell'errore da confronti multipli e replicazione multi-modello.
