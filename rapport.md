# Tobacco trial classification report

##### Robin Niel

## 1) Description du problème 

### 1.1) Contexte

Dans les années 60, de plus en plus d'études commençaient à montrer les effets nocifs du tabac sur la santé. Afin de limiter la perte de profit liée à la publication de ces études, les fabricants de tabac décidèrent de s'organiser et de créer une campagne nationale afin d'influencer l'opinion publique sur le sujet du tabac. S'en étant aperçu, l'État Américain décida de poursuivre en justice ces entreprises. Dans le cadre de ce procès, plus de 14 millions de documents ont été collectés et numérisés. Notre tâche pour ce travail pratique est de créer un modèle d'apprentissage automatique capable d'associer cesdits documents à leur catégorie. 

### 1.2) Les données

Pour ce travail pratique, nous avons à disposition 3482 documents répartis dans 10 différentes catégories, qui sont les suivantes (ici en anglais, comme dans les données) :

* Advertisement 

* Email 

* Form 

* Letter 

* Memo 

* Opening News 

* Note 

* Report 

* Resume 

* Scientific 

Nous pouvons tracer un camembert pour visualiser la distribution des classes dans les données. Voilà le graphique : 

![alt text](https://github.com/Rouen-NLP/final-lab-Datavoore/blob/master/Camenbert.png "Diagramme camembert fréquences classes")

Nous pouvons voir que le déséquilibre des données est léger, même si 3 classes sont sur-représentées : Memo, Letter et Email. Ce déséquilibre est logique par rapport au contexte de l'acquisition des données. Il ne faudra donc pas se fier qu'a la précision car celle-ci peut être biaisée par rapport à la répartition des classes. C'est pour celà que lors de la présentation des résultats, nous analyserons le rappel, la précision, le score f1 ainsi que la matrice de confusion afin d'être sûr que les erreurs sont bien réparties au niveau des classes. 

Un autre point important concernant les données est le fait que le texte dont nous allons nous servir à été récupéré de façon automique sur les documents numérisés. Celà peut induire certaines erreurs dans le texte. On peut lister quelques phrases contenant des erreurs extraites du jeu de données :

*  "The summary “haat of Pek formulad ‘pequeated by the Product pevelopuent comtécce "

*  "It. Wow business will be entertained from conunittee members following discussion of the preceding “tems."

Ainsi certaines erreurs seront inhérentes aux données et non pas au classifieur.

Au vu des ces erreurs mais aussi au fait que les données contiennent aussi beaucoup de noms propres et de vocabulaires spécifiques, je n'utiliserais pas d'embedding externes tels que word2vec. 

## 2) Résolution du problème

### 2.1) Etablissement d'une baseline 

Afin de rendre compte de la difficulté du problème et pouvoir présenter des améliorations, nous allons utiliser une méthode classique pour la classification de texte : Bayes naïf. Une méthode classique est d'abord de pré-traiter les données en enlevant la ponctuation. Ensuite afin d'obtenir une représentation vectorielle des données, comme dit précedemment je n'utiliserais pas d'embedding externe mais la représentation par tokens dans lequel chaque document est représenté par un vecteur d'une taille de **75956** (taille du vocabulaire) avec des 1 dans les composantes correspondants aux mots présents dans le document. 
Après avoir réalisé cette transformation, on peut entraîner le modèle Baiyes naïf sur des données d'entraînement (x_\train\_counts) et on le teste sur un jeu de test (x\_test\_counts). 
Voilà les résultats :

![alt text](https://github.com/Rouen-NLP/final-lab-Datavoore/blob/master/Results_Bayes.png "Résultats Bayes")

Nous pouvons voir que le score f1 est de 70%. On peut remarquer un certain déséquilibre dans la matrice de confusion, les classes les moins représentées semblent être souvent classifiées dans les classes les olus représentées. Par exemple la classe Note n'as eu aucun de ses représentants bien classifiés. 
Ces résultats permettent de dire que la tâche n'est pas trop complexe puisque la prédiction aléatoire donnerait aux alentours de 10% de précision et que Bayes atteint quasimment 70%. Toutefois il est possible d'améliorer notre classifieur. 
Une méthode donnant de bon résultats en NLP dernièrement sont les réseaux de neurones. Nous allons donc tenter d'appliquer un réseau de neurones au mêmes données afin de voir si une amélioration est observée. 

### 2.2) Amélioration par réseau de neurones

J'ai choisi une architecture dense classique que j'ai essayé de conserver peu profonde dû au manque de données. Il y aura donc une couche dense contenant 512 neurones en entrée reliées à une couche de décision avec une fonction d'activation softmax. On prendra la loss categorical_crossentropy de keras. On gardera le modèle avec la loss la plus basse. Voilà la courbe d'apprentissage du modèle, sur laquelle le modèle conservé est indiqué par une étoile :

![alt text](https://github.com/Rouen-NLP/final-lab-Datavoore/blob/master/Graph_learning.png "Courbe d'apprentissage du réseau")

On peut voir que du sur-apprentissage apparaît à partir de l'epoch ~10. On gardera le modèle de l'epoch 5 car c'est celui avec la loss la plus basse. 
On peut tester le modèle sur le même jeu de test que celui pour Bayes. Voilà les résultats avec les mêmes métriques. 

![alt text](https://github.com/Rouen-NLP/final-lab-Datavoore/blob/master/Results_NN.png "Résultats Neural network")

Par rapport au Bayes naïf le gain de performance est notable (~15%). On peut aussi remarquer que la matrice de confusion est beaucoup plus équilibrée. Les quelques erreurs sont bien réparties entre les classes.
On peut conclure que l'architecture utilisant un réseau de neurones est plus adaptée à ce problème que l'utilisation d'un Bayes naïf.

### 2.3) Conclusion

Lors de la partie précédente, nous avons pû voir que nous avons réussi à bien améliorer les performances par rapport à la baseline posée par l'utilisation Bayes naïf. Toutefois il est encore possible de gagner en perfomance.
Une première manière serait d'améliorer les données grâce à, par exemple, l'utilisation d'un modèle de langue afin de traiter les erreurs de transcription automatique décrites dans la partie 1.2. 
Il est aussi possible de prendre en compte les images numérisées des documents grâce à l'utilisation d'SVMs ou de réseaux de neurones convolutionnels afin d'ajouter des caractéristiques en entrée du classifieur final.
Enfin, il est possible de ne pas utiliser l'approche bag of words et de d'utiliser l'aspect séquentiel des documents en utilisant notamment des réseaux récurrents comme ceux utilisant les LSTMs. 
 
