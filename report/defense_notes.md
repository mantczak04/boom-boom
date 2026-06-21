# Notatki do obrony

1. **Co jest stanem MDP?** Tensor `H × W × C`: informacja o odkryciu, wartość
   odkrytego pola, kanał flag oraz — w trybie `state+prob` — prawdopodobieństwo miny.
2. **Co jest akcją?** Indeks pola od `0` do `H·W-1`, liczony wierszami.
3. **Jaka jest nagroda?** Bezpieczny ruch: `1-p`; mina: `-p`; wygrana:
   `1-p+B`; ponowne odkrycie: `0`.
4. **Dlaczego środowisko jest stochastyczne?** Przy resecie wyniki pól są losowane
   z `Bernoulli(p_mine)`. Ten sam wybór może mieć inny wynik w innym epizodzie.
5. **Co robi `action_mask`?** Oznacza dozwolone akcje i uniemożliwia agentom wybór
   odkrytych pól.
6. **Jak powstają prawdopodobieństwa?** Rozkład stały używa jednego `p`, jednostajny
   losuje z `[low, high]`, a skorelowany wygładza szum filtrem Gaussa.
7. **Random a Min-risk?** Random losuje dozwolone pole. Min-risk deterministycznie
   wybiera najmniejsze jawne `p_mine`.
8. **Co jest frontendem?** `app.py` w Streamlit: gra, sterowanie agentem, benchmark i
   opis modelu.
9. **Co jest backendem?** Pakiet `prob_minesweeper`: plansza, rozkłady, nagrody,
   środowisko Gymnasium, agenci i ewaluacja.
10. **Co dodano?** Grywalny interfejs, agentów wielokrotnego użytku, porównywalną
    ewaluację, testy integracyjne oraz dokumentację do prezentacji.

Projekt pozostaje w obszarze RL, choć dwaj dołączeni agenci są bazami
heurystycznymi, a nie wytrenowanymi sieciami. Ich rolą jest zapewnienie mierzalnego
punktu odniesienia dla kolejnego modelu RL.
