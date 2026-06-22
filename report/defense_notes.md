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
    nie używa `action_mask` podczas treningu. Podczas demonstracji i ewaluacji błędna
    akcja jest zastępowana losową akcją dozwoloną. MaskablePPO używa maski akcji
    bezpośrednio podczas uczenia i predykcji.
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
24. **Dlaczego dodano MaskablePPO?** Ponieważ podstawowy DQN ze Stable-Baselines3
    nie używa `action_mask` podczas treningu. MaskablePPO obsługuje maskowanie akcji,
    więc lepiej pasuje do gry, w której część pól staje się niedozwolona.
25. **Czym różni się DQN od MaskablePPO?** DQN uczy funkcji wartości akcji `Q(s,a)`,
    a PPO uczy bezpośrednio polityki wyboru akcji. MaskablePPO dodatkowo zeruje
    prawdopodobieństwo wyboru akcji niedozwolonych.
26. **Czy Min-risk jest uczciwym baseline'em?** Nie w hidden-risk. Min-risk korzysta
    z ukrytego `p_mine`, którego modele RL w trybie `state` nie obserwują. Dlatego
    jest oracle/reference, a nie fair baseline.
27. **Po co mierzyć invalid-action rate DQN?** To pokazuje, jak często DQN wybiera
    formalnie istniejącą, ale niedozwoloną akcję, czyli odkryte pole. MaskablePPO
    rozwiązuje ten problem przez maskowanie podczas uczenia i predykcji.
28. **Dlaczego są dwa reżimy eksperymentów?** Łatwiejszy reżim sprawdza, czy agent
    potrafi nauczyć się użytecznej polityki, a trudny reżim hidden-risk pokazuje
    ograniczenia i trudność oryginalnego środowiska.

Random jest bazą, Min-risk jest oracle, a DQN i MaskablePPO są trenowanymi modelami
RL. Wyniki należy porównywać przy tych samych rozmiarach planszy, rozkładach, trybach
i seedach.
