## Test des Commentaires Notion

Ce fichier contient des exemples de commentaires Notion mal formatés qui génèrent des ` ` bizarres.

### Exemples de Problèmes

#### Commentaire standalone mal formaté
[^comment]: Ceci est un commentaire qui devrait être converti en note de bas de page

#### Commentaire inline mal formaté
Voici du texte normal avec un commentaire inline[^comment] qui traînent dans le contenu.

#### Formatage gras corrompu *Texte en gras * qui a été corrompu par les commentaires.

[^comment] qui devrait rester en gras.

#### Cas qui ne doivent PAS être modifiés TITRE EN GRAS - Ceci doit rester en gras car c'est un titre.

[^comment] - Ceci doit rester en gras car c'est court et important.

[Lien](https://example.com) - Ceci contient des liens et ne doit pas être modifié. 123 - Ceci ne doit pas être modifié car c'est juste des chiffres.

### Résultat Attendu

Après traitement, les commentaires devraient être convertis en notes de bas de page :

- `[^comment]` → `[^comment]: Ceci est un commentaire`
- `texte [^comment] suite` → `texte[^comment] suite`
- Les astérisques orphelins devraient être supprimés
- Le formatage gras légitime devrait être préservé
