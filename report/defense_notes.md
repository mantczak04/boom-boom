# Notatki do obrony

1. **Co jest stanem MDP?** Tensor `H × W × C`: informacja o odkryciu, wartość
   odkrytego pola, kanał flag oraz — w trybie `state+prob` — prawdopodobieństwo miny.
2. **Co jest akcją?** Indeks pola od `0` do `H·W-1`, liczony wierszami.
3. **Jaka jest nagroda?** W hidden-risk: bezpieczny ruch `+0,1`, mina `-1`,
   dodatkowy bonus za wygraną `+10`, ponowne odkrycie `0`. Pierwotna nagroda
   risk-adjusted nadal jest dostępna.
4. **Dlaczego środowisko jest stochastyczne?** Przy resecie wyniki pól są losowane
   z `Bernoulli(p_mine)`. Ten sam wybór może mieć inny wynik w innym epizodzie.
5. **Co robi `action_mask`?** Oznacza dozwolone akcje i uniemożliwia agentom wybór
   odkrytych pól.
6. **Jak powstają prawdopodobieństwa?** Rozkład stały używa jednego `p`, jednostajny
   losuje z `[low, high]`, a skorelowany wygładza szum filtrem Gaussa.
7. **Random a Min-risk?** Random losuje dozwolone pole. Min-risk deterministycznie
   wybiera najmniejsze `p_mine`, które w hidden-risk jest informacją ukrytą.
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
17. **Dlaczego dodano hidden-risk?** Jawne `p_mine` sprowadzało problem głównie do
    lokalnego wyboru najmniejszego ryzyka. Ukrycie prawdopodobieństw wymusza użycie
    informacji z kolejnych odkryć.
18. **Dlaczego MinRiskAgent jest oracle?** Czyta `p_mine` bezpośrednio z planszy,
    chociaż kanał ten nie występuje w obserwacji DQN `obs_mode=state`.
19. **Co obserwuje DQN?** Maskę odkrytych pól, wartości wskazówek i zarezerwowany
    kanał flag. Nie obserwuje pola prawdopodobieństw.
20. **Co oznacza wskazówka `actual_count`?** Dokładną liczbę rzeczywiście
    wylosowanych min w ośmiu sąsiednich polach, bez pola centralnego.
21. **Dlaczego problem jest sekwencyjny?** Odkrycie wpływa nie tylko na bieżącą
    nagrodę, ale ujawnia wskazówkę użyteczną przy następnych decyzjach.
22. **Po co początkowy blok 2×2?** Daje agentowi cztery wskazówki przed pierwszą
    decyzją, ograniczając epizody kończące się natychmiastowym losowym trafieniem.
    Automatyczne odkrycia nie dają nagrody i nie zużywają kroków.
23. **Jak blok jest bezpieczny?** Po wylosowaniu losowej pozycji cztery wyniki min
    w bloku są wymuszane jako bezpieczne. Jedno dodatkowe ukryte pole także jest
    bezpieczne, aby epizod nie był wygrany już przy resecie.

Random jest bazą, Min-risk jest oracle, a DQN jest trenowanym modelem RL. Wyniki
należy porównywać przy tych samych rozmiarach planszy, rozkładach, trybach i seedach.
