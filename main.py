from scalability.resolution_scalability import run_scalability

if __name__ == '__main__':
    run_scalability(100)
    run_scalability(108)
    run_scalability(116)
    run_scalability(124)
    run_scalability(132)
    run_scalability(140)
    run_scalability(148)
    run_scalability(156)



# todo creare una classe results che tenga il dataframe e la matrice di similarità (...)
# todo pensare alla metrica di bontà dei clusters  ---->  IN TEORIA FATTO, DA PROVARE (VEDERE SOTTO)
# todo provare a creare i clusters con od random (prese da qualsiasi quadrante) e quindi vedere il valore degli indici di similarità. Dovrebbero essere vicino lo zero
# todo mettere restric_paths_to_quadrant a False e fare le stesse prove (magari direttamente con 10 istanze)

# todo fare altre 5 prove sui test già fatti, recuperare il seed e aumentarlo
# todo vedere se ha senso parallelizzare il calcolo degli indici di bontà dei clusters (non penso...)
# todo pensare alla metrica di sovrapposizione spazio (spazio-temporale) dentro i clusters