# Probabilistyczny Saper jako środowisko uczenia ze wzmocnieniem

## 1. Cel projektu

Celem jest aplikacja prezentująca probabilistyczny wariant Sapera jako środowisko
uczenia ze wzmocnieniem (RL). Projekt łączy backend zgodny z Gymnasium, interfejs
Streamlit oraz porównanie agentów bazowych z modelami DQN i MaskablePPO.

## 2. Opis problemu

Każde pole ma prawdopodobieństwo miny $p_{r,c}$. Na początku epizodu ukryty wynik
każdego pola jest losowany z rozkładu Bernoulliego. W głównym wariancie hidden-risk
agent DQN nie widzi tych prawdopodobieństw. Po bezpiecznym odkryciu poznaje rzeczywistą
liczbę min w sąsiedztwie. Wygrywa po odkryciu wszystkich bezpiecznych pól, a przegrywa
po odkryciu miny. Pierwotny wariant z jawnym $p_{r,c}$ pozostaje dostępny.

## Wariant hidden-risk i uzasadnienie RL

W pierwotnym wariancie `state+prob` agent znał prawdopodobieństwa min, przez co
heurystyka MinRisk była bardzo silna. W wariancie hidden-risk prawdopodobieństwa są
ukryte, a odsłonięcie pola dostarcza informacji w postaci rzeczywistej liczby min w
sąsiedztwie. Decyzja agenta wpływa więc zarówno na nagrodę natychmiastową, jak i na
informację dostępną w kolejnych stanach. Jest to sekwencyjny problem decyzyjny w
warunkach niepewności.

Każdy epizod rozpoczyna się od losowego, bezpiecznego bloku $2\times2$, który jest
automatycznie odkryty bez nagrody i bez zwiększania licznika kroków. DQN otrzymuje
więc cztery rzeczywiste wskazówki przed pierwszą decyzją, zamiast wykonywać ruch bez
żadnej informacji. Dodatkowe pole poza blokiem pozostaje ukryte, ale jest wymuszane
jako bezpieczne, aby reset nie zwracał już ukończonej planszy.

## 3. Model matematyczny

Problem opisujemy jako MDP $(S,A,P,R,\gamma)$:

- $S$ — stan planszy: maska odkrytych pól, wartości odkrytych pól, kanał flag
  (zarezerwowany) i opcjonalny kanał prawdopodobieństw;
- $A=\{0,\ldots,H\cdot W-1\}$ — wybór pola w porządku wierszowym;
- $P$ — przejście ujawniające wcześniej wylosowany wynik pola;
- $R$ — funkcja nagrody skupiona na ukończeniu planszy;
- $\gamma$ — współczynnik dyskontowania przyszłych nagród.

$$
r_t = \begin{cases}
0{,}1 & \text{bezpieczne odkrycie},\\
-1 & \text{trafienie miny},\\
10{,}1 & \text{ruch wygrywający},\\
0 & \text{brak zmiany}.
\end{cases}
$$

Dla bezpiecznego pola $i$ wskazówka ma postać
$c_i=\sum_{j\in N(i)}M_j$, gdzie $M_j\sim\mathrm{Bernoulli}(p_j)$.

Celem agenta jest maksymalizacja
$\mathbb{E}[\sum_{t=0}^{T}\gamma^t r_t]$.

## 4. Architektura aplikacji

Frontend `app.py` wykorzystuje Streamlit i udostępnia zakładki Gra, Benchmark oraz
Model. Backend `prob_minesweeper` zawiera logikę planszy, rozkłady, środowisko
Gymnasium, nagrody, agentów i ewaluację. Stan bieżącej gry jest przechowywany w
`st.session_state`.

## 5. Implementacja

Rozkład `constant` przypisuje wszystkim polom jedno $p$. `uniform` losuje każde $p$
niezależnie z zadanego przedziału. `correlated` wygładza biały szum filtrem
Gaussa, tworząc przestrzennie skorelowane obszary ryzyka. `action_mask` blokuje pola
już odkryte. Testy sprawdzają logikę domenową, API Gymnasium, agentów i ewaluację.

## 6. Agenci i metody porównawcze

`RandomAgent` losuje jednostajnie spośród dozwolonych akcji. `MinRiskAgent` wybiera
dozwolone pole o najmniejszym $p_{r,c}$; przy remisie wybiera pierwsze. W hidden-risk
jest to oracle z uprzywilejowanym dostępem do ukrytej informacji, a nie uczciwa baza
dla modeli RL. DQN, MaskablePPO i Random korzystają z widocznego stanu.

## Deep Q-Network jako wykorzystany model RL

W projekcie wykorzystano gotową implementację DQN z biblioteki Stable-Baselines3.
DQN pasuje do środowiska, ponieważ przestrzeń akcji jest dyskretna: każda akcja jest
indeksem jednego pola. Sieć uczy się funkcji wartości akcji $Q(s,a)$, czyli oczekiwanej
długoterminowej wartości odkrycia pola $a$ w stanie $s$. Celem uczenia jest przybliżenie
celu Bellmana

$$
y = r + \gamma \max_{a'} Q(s',a'),
$$

przy użyciu błędu $(y-Q_\theta(s,a))^2$. Tensor planszy $H\times W\times C$ jest przed
podaniem do polityki MLP spłaszczany do wektora. W odróżnieniu od `MinRiskAgent`, który
stosuje stałą lokalną regułę minimalizacji bieżącego $p_{r,c}$, DQN uczy się na podstawie
interakcji i uwzględnia zdyskontowane przyszłe nagrody.

Stable-Baselines3 DQN nie wykorzystuje automatycznie `action_mask` podczas treningu.
Podczas ewaluacji i demonstracji przewidziana już odkryta akcja jest zastępowana
losową akcją dozwoloną. Dla DQN raportujemy `invalid-action rate`, czyli odsetek
predykcji wskazujących pole już odkryte. Model jest trenowany i oceniany na planszy
o tym samym rozmiarze.

## MaskablePPO jako model z maskowaniem akcji

Drugim modelem RL jest MaskablePPO z pakietu `sb3-contrib`. W przeciwieństwie do
podstawowego DQN, MaskablePPO wykorzystuje maskę dozwolonych akcji podczas uczenia
i predykcji. W naszym środowisku maska blokuje pola już odkryte, więc agent nie traci
kroków treningowych na akcje, które nie zmieniają stanu planszy.

PPO uczy polityki stochastycznej bezpośrednio, a nie funkcji wartości każdej akcji
tak jak DQN. Wariant z maską lepiej pasuje do Sapera, ponieważ liczba formalnie
dostępnych akcji jest stała, ale część akcji staje się niedozwolona po odkryciu pól.

## 7. Eksperymenty

Agenci są oceniani na tych samych konfiguracjach i sekwencjach seedów. Rejestrujemy
liczbę zwycięstw, porażek i uciętych epizodów, średnią nagrodę, średnią liczbę
kroków oraz `invalid-action rate` dla DQN. Raport rozdziela dwa reżimy: łatwiejszy
test uczenia i trudny stress test hidden-risk.

## 8. Wyniki

### Łatwiejszy reżim uczenia: `constant p=0.15`

| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Min-risk (oracle) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| DQN (random fallback) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| MaskablePPO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

DQN invalid-action rate: TODO

### Trudny stress test hidden-risk: `correlated`

| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Min-risk (oracle) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| DQN (random fallback) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| MaskablePPO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

DQN invalid-action rate: TODO

Łatwiejszy reżim pokazuje, czy pipeline RL uczy się użytecznej polityki w warunkach,
w których gra jest realnie wygrywalna. Trudny reżim `correlated` traktujemy jako
stress test. Jeśli nawet Min-risk oracle ma niski win rate, to średnia nagroda,
średnia liczba kroków i invalid-action rate DQN są bardziej informacyjne niż sam
win rate. Zrzuty interfejsu należy dodać do `report/screenshots/` przed oddaniem.

## 9. Wnioski

Porównanie pokazuje wpływ uprzywilejowanej informacji o ryzyku i uczenia wartości
przyszłych nagród na strategię. `MinRiskAgent` może osiągać lepsze wyniki, ponieważ
bezpośrednio korzysta z ukrytego $p_{r,c}$ jako oracle. DQN pozostaje jednak polityką
wyuczoną, a nie z góry zapisaną regułą. MaskablePPO jest czystszą metodą RL dla tego
środowiska, ponieważ korzysta z maski akcji w samym algorytmie. Nie zakładamy jednak,
że musi pokonać oracle Min-risk. Ograniczeniami są niestabilność treningu i zależność
modeli od rozmiaru planszy.

## 10. Instrukcja uruchomienia

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_dqn.py --timesteps 500000
uv run python experiments/evaluate_dqn.py --episodes 500
uv run python experiments/train_maskable_ppo.py --timesteps 100000
uv run python experiments/evaluate_maskable_ppo.py --episodes 500
uv run python experiments/compare_rl_agents.py --episodes 500
uv run streamlit run app.py
```
