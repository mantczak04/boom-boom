# Probabilistyczny Saper jako środowisko uczenia ze wzmocnieniem

## 1. Cel projektu

Celem jest aplikacja prezentująca probabilistyczny wariant Sapera jako środowisko
uczenia ze wzmocnieniem (RL). Projekt łączy backend zgodny z Gymnasium, interfejs
Streamlit oraz porównanie dwóch agentów bazowych z wytrenowanym agentem DQN.

## 2. Opis problemu

Każde pole ma znane prawdopodobieństwo miny $p_{r,c}$. Na początku epizodu
ukryty wynik każdego pola jest losowany z rozkładu Bernoulliego. Agent widzi
prawdopodobieństwa, ale poznaje wynik dopiero po odkryciu pola. Wygrywa po odkryciu
wszystkich bezpiecznych pól, a przegrywa po odkryciu miny.

## 3. Model matematyczny

Problem opisujemy jako MDP $(S,A,P,R,\gamma)$:

- $S$ — stan planszy: maska odkrytych pól, wartości odkrytych pól, kanał flag
  (zarezerwowany) i opcjonalny kanał prawdopodobieństw;
- $A=\{0,\ldots,H\cdot W-1\}$ — wybór pola w porządku wierszowym;
- $P$ — przejście ujawniające wcześniej wylosowany wynik pola;
- $R$ — funkcja nagrody zależna od ryzyka i wyniku ruchu;
- $\gamma$ — współczynnik dyskontowania przyszłych nagród.

$$
r_t = \begin{cases}
1-p_a & \text{bezpieczne odkrycie},\\
-p_a & \text{trafienie miny},\\
1-p_a+B & \text{ruch wygrywający},\\
0 & \text{brak zmiany}.
\end{cases}
$$

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
dozwolone pole o najmniejszym $p_{r,c}$; przy remisie wybiera pierwsze. Obaj agenci są
punktami odniesienia dla wytrenowanego modelu DQN.

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
kroków. Przed oddaniem należy podać rozmiar planszy, rozkład, parametry, seed i
liczbę epizodów.

## 8. Wyniki

| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Min-risk | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| DQN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Przed oddaniem należy wkleić wyniki z zakładki Benchmark oraz zrzuty interfejsu
do `report/screenshots/`.

## 9. Wnioski

Porównanie pokazuje wpływ jawnej informacji o ryzyku i uczenia wartości przyszłych
nagród na strategię. `MinRiskAgent` może osiągać lepsze wyniki, ponieważ bezpośrednio
korzysta z $p_{r,c}$ i jest silną heurystyką lokalną. DQN pozostaje jednak polityką
wyuczoną, a nie z góry zapisaną regułą. Ograniczeniami są niestabilność treningu,
brak maskowania akcji w samym algorytmie oraz zależność modelu od rozmiaru planszy.

## 10. Instrukcja uruchomienia

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_dqn.py --timesteps 50000
uv run python experiments/evaluate_dqn.py --episodes 100
uv run streamlit run app.py
```
