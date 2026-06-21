# Probabilistyczny Saper jako środowisko uczenia ze wzmocnieniem

## 1. Cel projektu

Celem jest aplikacja prezentująca probabilistyczny wariant Sapera jako środowisko
uczenia ze wzmocnieniem (RL). Projekt łączy backend zgodny z Gymnasium, interfejs
Streamlit oraz porównanie dwóch agentów bazowych z wytrenowanym agentem DQN.

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
dla DQN. DQN i Random korzystają z widocznego stanu.

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
dozwoloną akcją `MinRiskAgent`; awaryjnie losowana jest dowolna akcja dozwolona. Model
jest trenowany i oceniany na planszy o tym samym rozmiarze.

## 7. Eksperymenty

Agenci są oceniani na tych samych konfiguracjach i sekwencjach seedów. Rejestrujemy
liczbę zwycięstw, porażek i uciętych epizodów, średnią nagrodę oraz średnią liczbę
kroków. Poniższy eksperyment używa planszy $5\times5$, rozkładu `correlated`,
`obs_mode=state`, `clue_mode=actual_count`, `initial_reveal=safe_2x2`, nagrody
`completion`, seedu początkowego 123 i 500 epizodów. Model DQN trenowano przez
500 000 kroków z seedem 42.

## 8. Wyniki

| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 500 | 0 | 500 | 0 | 0,0% | -0,8806 | 2,194 |
| DQN 500k | 500 | 4 | 496 | 0 | 0,8% | -0,4776 | 5,336 |
| Min-risk (oracle) | 500 | 3 | 497 | 0 | 0,6% | -0,4316 | 6,018 |

DQN osiągnął wyższą średnią nagrodę i więcej zwycięstw niż Random oraz o jedno
zwycięstwo więcej niż Min-risk. Oracle zachował nieco wyższą średnią nagrodę dzięki
dostępowi do ukrytego $p_{r,c}$. Niskie bezwzględne win rate pokazuje trudność
wariantu i pozostawia miejsce na dłuższy trening lub algorytm wykorzystujący
maskowanie akcji. Zrzuty interfejsu należy dodać do `report/screenshots/` przed
oddaniem.

## 9. Wnioski

Porównanie pokazuje wpływ uprzywilejowanej informacji o ryzyku i uczenia wartości
przyszłych nagród na strategię. `MinRiskAgent` może osiągać lepsze wyniki, ponieważ
bezpośrednio korzysta z ukrytego $p_{r,c}$ jako oracle. DQN pozostaje jednak polityką
wyuczoną, a nie z góry zapisaną regułą. Ograniczeniami są niestabilność treningu,
brak maskowania akcji w samym algorytmie oraz zależność modelu od rozmiaru planszy.

## 10. Instrukcja uruchomienia

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_dqn.py --timesteps 500000
uv run python experiments/evaluate_dqn.py --episodes 500
uv run streamlit run app.py
```
