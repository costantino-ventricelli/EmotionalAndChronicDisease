# EmotionalAndChronicDisease
Individuazione correlazioni tra stati emotivi alterati e patologie croniche, attraverso studi del tratto biometrico con tecniche di shallow e deep learning.

La struttura del codice è la seguente:

    - il package DatasetManager contiente le classi che permettono di gestire l'intero dataset "HAND" e i task presenti
    al suo interno:
    
        - HandManager.py gestisce appunto il contenuto fisico del dataset, permette di leggere i dati e ottenere 
            le informazioni fondamentali;
        - TaskManager.py gestisce le operazioni sui task permettendo di ottenere i task necessari agli studi tramite
            le varie funzioni;
        - Costants.py contiene tutte le costanti utilizzate per identificare i task all'interno del progetto,
            le costanti che identificano i tratti in aria o in superficie per lo shallow learning, e le associazioni
            tra gli alias dei task e i loro nomi completi.
            
    - il pakage DeepLearningClassifier contiente tutti i file necessari per gestire il classificatore neurale:
        - DeepKFoldValidation.py permette di avviare analisi con metodo di valutazione k-fold validation, l'interfaccia
            si compone di due metodi il costruttore e il metodo che avvia la valutazione, al costruttore vanno passati nell'
            ordine: gli id degli utenti sani, gli id degli utenti mild e gli id degli id disease, successivamente va passato
            il dataset viene passato in forma di dizionario e infine vengono richiesti arbitrariamente il numero di fold, se 
            il valore viene omesso verrà effettuata una 10 fold validation, un esempio di utilizzo in questa classe è fornito
            in Experiment.Experiment_15.py;
        - DeepLeaveOneOut.py permette di effettuare una valutazione di tipo leave one out passando come valori al costruttore:
            il numero minimo di campioni che i files devono possedere per essere presi in considerazione, 
            la lunghezza che ogni campione assumerà nel tensore, l'estrattore delle feature: deve restituire il tensore che 
            verrà utilizzato dalla rete per la modellazione e il test, ad esempio RHSDistanceExtraction o RTPExtraction, 
            il numero di fearures che si vogliono utilizzare, ad esempio RHS ha 3 features mentre RTP ne ha 5, il dataset
        - MachineLearningModel.py contiente la definizione della rete e i vari metodi per addestrarla, testarla e valutarne
            le performance.
        - MachineLearningModelAttentionLayer.py contiente anch'essa un modello di rete che include un layer con meccanismo
            di attenzione, vedi la tesi.
        - RHSDistanceExtraction.py permette di generare i tensori necessari al modello bilanciandoli sulle due classi in 
            automatico, i commenti e gli svariati esempi presenti dall'esperimento 1 all'esperimento 8 forniscono linee guida 
            su come utilizzare questa classe;
        - RTPExtraction permette di ottenre gli stessi risultati della classe precedente però modificando il tensore il quale
            passa a 4 dimensioni invece di 3 aggiungendo il new_stroke_flag alle 3 precedenti (RSH_x, RHS_y, PS)
    - il package Experiment contiene i modelli generali degli esperimenti, in quanto molti di quelli che ho fatto io erano
        pressoché identici tra di loro, variava solo la selezione dei task, questi forniscono un modello generale il quale poi
        viene richiamato dai vari script presenti in src/ che ogni volta configurano l'esperimento per ottenere i risultati
        richiesti.
    - il package ShallowLearningClassifier contiene tutte le classi necessarie all'anaisi con classificatori superficiali:
        RF e SVM:
        - CreateDictDataset.py permette di creare una dizionario python che contiene tutto il dataset HAND il funzionamento
            di questa classe è esplicato nella stessa assieme alla struttura dati che verrà generata.
        - Features.py contiene tutti i metodi che permettono di estrarre molteplici features di tipo verroriale e scalare,
            il nome di ogni metodo è autodesctittivo, inoltre ognuno dei metodi è accompagnato da una descrizione che spiega
            la feature che quel metodo calcola;
        - FeaturesExtraction.py è la classe che combinata a Features.py descrive nel dettaglio le features che estraee
            il suo utilizzo permette di generare il dataset di 3092 features presentato nella tesi, dando in pasto ad essa
            solo il dataset, il funzionamento di questa classe è esplicato nella classe stessa, di fatto per modificare le
            features che vengono estratte "basterà" modificare i files a cui la classe fa riferimento: feature_list.csv e 
            header_file.csv il primo contiene una lista di tutte le features globali che verranno estratte, ad esempio ci 
            dice che verrà calcolata la derivata frazionaria in superficie, in aria, sui vari assi e per le varie funzioni
            mentre header file contiene una lista di tutti i valori che poi ci ritroveremo, quindi di tutte le funzioni 
            statistiche che vengono applicate alle feature funzionali, eliminando i guppi di features da questi file esse
            non verranno più calcolate.
        - FeatureManager.py permette di astrarre ulteriormente il processo complicato e laborioso di estrazione, di fatto da
            questa classe è possibile avviare la vera e propria estrazione, non ha bisogno di parametri di configurazione, 
            al termine genererà una cartella detta features in resources la quale contiene 23 file ognuno dei quali contiene
            in ogni riga la lista delle features estratte per ogni paziente, il collegamento riga paziente si mantiene nel
            file patients.txt, che memorizza l'ordine con cui sono state estratte le features per ogni paziente.
        - FeatureSelection.py contiene il RandomForest che permette di estrarre il set di features migliori, utilizza il 
            grid search per l'ottimizzazione degli iperparametri;
        - ShallowLeaveOneOut.py permette di avviare una validazione leve one out utilizzando in coppia il random forest e
            la support vector machine;
        - SVCModel.py permette di addestrare e testare la SVM utilizzando il grid search per ottimizzare gli iperparametri.
    - nella cartella src invece trovate tutti gli script che descrivono le configurazioni di tutti gli esperimenti che ho 
        fatto, per avviarli vi basterà installare tutte le librerie richieste e avviarli da console, consiglio di non farlo
        alla leggera, in quanto molti degli esperimenti sono davvero pesanti e lunghi leggete cosa l'esperimento si propone
        di fare prima di avviarlo.

Buona forutuna
     
# Librerie necessarie:
  - tensorflow;
  - keras;
  - matplotlib;
  - numpy;
  - EMD-signal;
  - sklearn;
  - differint;
  - pandas;
  - playsound.