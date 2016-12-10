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

per fare il downsample delle immagini. Questo genera due pickle nella cartella `features/` che sono due dict con dentro gli array numpy 3x3x3 downsampled. Hai tempo di farti un caffettino.

Poi:

```
python classifier.py
```

Addestra i classificatori (uno che performava bene su tutte le labelle è il processo gaussiano) e tira giù due statistiche
