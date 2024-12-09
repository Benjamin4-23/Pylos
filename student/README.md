# ai_pylos met DQN

Logging:
    BenjaminPlayer (minimax) speelt met diepte 9 games tegen minimax4
    
    Tijdens het spelen logt hij zijn uitgevoerde zetten als (boardstate,action,reward) naar een csv file.
    
    Boardstate: string met 30 karakters, 1 voor elke locatie. 0 is leeg, 1 is zijn sphere, 2 is opponent zijn sphere.
    
    Action: string met 304 karakters, 1 voor elke actie. Er zijn meer acties mogelijk maar veel zijn niet valid. 
            Move acties: 0-207 (level 0 to 1 or 2), 208-243 (level 1 to 2)
            Place acties: 244-272  (1 voor elke locatie)
            Remove acties: 273-302 (1 voor elke locatie) 
            Pass actie: 303 
            Alle karakters zijn 0 behalve de actie die hij uitvoert, die is 1.
    
    Reward: zelfgemaakte reward functie op basis van reserve spheres, blockable squares, blocking squares en completed squares.

Model training: 
    Maak 1 data file met remove/pass acties en 1 data file met move/place acties. (anders kan het zijn dat het model een remove actie voorspelt in domove, daarom 2 modellen)
    
    Train 2 modellen met deze data:
        Geef als input de boardstate en laat het model 304 qvalues voorspellen (1 voor elke actie)
        Leer het model (door loss functie te minimaliseren) om de qvalues die in de dataset zitten voor deze state zo hoog mogelijk te maken, dit zijn namelijk zetten die minimax 9 zou doen.
        Het model zal voor een bepaalde state de qvalues voorspellen voor alle acties, maar het zal de qvalues voor de acties die in de dataset zitten verhogen.

    We willen zien dat de qvalues voor de acties die in de dataset zitten zo hoog mogelijk zijn, en de qvalues voor de acties die niet in de dataset zitten zo laag mogelijk zijn.
    
    We proberen een marge te bekomen tussen de qvalues voor de acties die in de dataset zitten en de qvalues voor de acties die niet in de dataset zitten zodat we meerdere goede acties kunnen leren en hier 1 uit kunnen kiezen.
    
    Exporteer het model naar een onnx file.

Model gebruiken:
    Wanneer we het model willen gebruiken geven we het als input de boardstate en laat het model 304 qvalues voorspellen (1 voor elke actie)
    we kunnen dan de actie met de hoogste qvalue kiezen en die uitvoeren. Als die actie niet valid is proberen we de actie met 2de hoogste qvalue uit te voeren... etc.


Resultaten:
    Zie de resultaten folder voor de resultaten.

Mogelijke verbeteringen:
    Trainen op moves van een betere speler (minimax 15)
    training optimaliseren:
        Bij de 1k boardstates subset gaan we snel naar 99% selection accuracy (aantal keer dat het model de actie van minimax 9 voorspelt).
        bij de grote dataset (move/place dataset heeft 10m boardstates, remove/pass heeft 2.5M boardstates) gaan we na 1dag trainen maar tot 80-85% selection accuracy. Dit kan een lokaal minimum zijn waar we uit kunnne komen door noise/dropout te implementeren.
    Self play:
        We kunnen het model ook trainen op zetten die het model zelf maakt. Dit zal waarschijnlijk betere resultaten geven. (reward dan gebaseerd op gewonnen/veloren)
