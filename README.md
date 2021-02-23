# EmotionalAndChronicDisease
Individuazione correlazioni tra stati emotivi alterati e patologie croniche, attraverso studi del tratto biometrico con tecniche di shallow e deep learning.

Il package DatasetManager contiene tutti gli strumenti per accedere al dataset Hand e ottenere le informazioni necessarie
all'avvio delle analisi tramite modelli ad apprendimento profondo o superficiale:

- la classe FileManager.py contiene tutti i metodi che permettono di accedere alla struttura 
del dataset e ottenerne i dati, come percorsi, task, pazienti stato di salute etc...
  I commenti presenti nel codice esplicano il funzionamento dei metodi;
- la classe Costants.py contiene tutte le costanti ricorrenti nel codice.

Il pakage DeepLearningClassifier contiene tutte le classi che permettono la corretta gestione del modello ad apprendimento
profondo sviluppato per il lavoro di tesi:

- LeaveOneOutMethod.py è una classe che permette di tesare il modello utilizzando la metodologia leave one out.

    È possile specificare la lunghezza minima dei file da considerare, il numero di campioni che comporranno ogni riga del 
  tensore, la fearure da utilizzare: RHS o RTP (per il momento), il numero delle dimensioni che comporranno i tensori e
  il dataset da utilizzare;
- MachineLearningModel.py è la classe che costruisce la rete permette l'addestramento, la validazione e il test della rete.

    Fornisce inoltre le metriche di misurazione delle performance (accuracy, precision, recall, f_score) sotto forma di tupla
chiamando il metodo "evaluate_results". Ovviamente il codice è ampiamente commentato;
- RHSDistanceExtraction.py è la classe che permette di estrarre i segmenti RHS dal dataset.

    Ovviamente la classe ottiene tensori multidimensionali perfettamente bilanciati, nel caso in cui si forniscano in input ai metodi
molteplici file da cui estrarre i valori richiesti (extract_rhs_known_state). Nel caso in cui si vogliano estrarre dati da un signolo file, e quindi non
  è possibile bilanciare i dati allora si utilizzaerà un metodo più diretto che non manipola il bilanciamento del dataset 
  ma ottiene semplicemente il tensore richiesto (extract_rhs_file);
- RTPExtraction.py si comporta esattamanete come RHSDistanceExtraction, ma estrare dati di tipo Row Touch Point, che quindi
presentano il new_stroke flag tra le dimensioni del tensore;
- TaskManager.py è la classe che si occupa di gestire estrarre informazioni e dati dai file che campionano i vari task del 
dataset. Ovviamente anche in questo caso ci affidiamo al commento del codice per l'eviscerazione del funzionamento dei vari
  metodi presenti nella calsse.
  
Il package Experiment contiene i vari esperimenti che sono stati eseguiti nella per la stesura della tesi, ogni classe è 
ampiamente commentata è mostra come utilizzare i package sopra descritti per addestrare, validare e testare le reti con vari
approcci: LeaveOneOut, decisione iniziale prestabilita del numero dei file da usare nelle varie fasi etc...

Ogni esperimento viene richiamato dagli script semplici contenuti nella cartella src.

