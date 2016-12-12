# Computational Phrenology

1. Scan brains
2. Classify degenerates
3. Invade Poland
4. ???
5. Profit

## Procedura

Scaricare i dati del caggole in questa cartella, estrarre il tar gz.

Girare

```
python flattener.py
```

per fare l'estrazione delle features. Questo genera due pickle nella cartella `features/` che sono due dict con dentro gli array delle features. Come features ho usato i primi modi della DFT. Hai tempo di farti un caffettino.

Poi:

```
python classifier.py
```

Addestra i classificatori e tira giù due statistiche. Poi, neanche te ne accorgi, ma nel frattempo ti ha preparato anche il file da sottomettere per l'esame, che si chiama `submission`.
