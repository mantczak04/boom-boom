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
11. **Czego uczy się DQN?** Przybliża funkcję wartości akcji `Q(s,a)`, czyli
    oczekiwaną sumę zdyskontowanych nagród po wybraniu akcji `a` w stanie `s`.
12. **Dlaczego DQN jest odpowiedni?** Środowisko ma dyskretną przestrzeń `H·W` akcji,
    a DQN zwraca wartość Q dla każdej z nich. Obserwację planszy spłaszczamy dla MLP.
13. **Co oznacza `Q(s,a)`?** To oszacowanie długoterminowej jakości odkrycia pola `a`
    w bieżącym stanie, a nie tylko jego natychmiastowego ryzyka.
14. **Jaki jest cel Bellmana?** `y = r + γ max_a' Q(s',a')`. Uczenie minimalizuje
    kwadrat różnicy między `y` a bieżącym `Q(s,a)`.
15. **Jak obsługiwane jest maskowanie akcji?** Standardowy DQN ze Stable-Baselines3
    nie używa `action_mask` podczas treningu. Podczas demonstracji błędna akcja jest
    zastępowana dozwoloną akcją Min-risk, a w ostatniej kolejności losową dozwoloną.
16. **Dlaczego Min-risk może wygrać z DQN?** Ma bezpośredni dostęp do `p_mine` i jest
    silną heurystyką lokalną. DQN uczy się przez interakcję, więc przy małej liczbie
    kroków treningowych może gorzej przybliżać wartości. Nie musi wygrać, aby pokazać
    wyuczoną politykę RL i uczciwe porównanie metod.

Random i Min-risk są bazami, natomiast DQN jest trenowanym modelem RL. Wyniki należy
porównywać przy tych samych rozmiarach planszy, rozkładach i seedach.
