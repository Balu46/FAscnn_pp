# Notatki: eksperymenty dla architektur FAscnn_pp

## Przyjete kryterium oceny

Baseline odniesienia: `FastSCNN`

- `mIoU`: `0.641077`
- `Pixel Accuracy`: `0.939707`
- `GPU FPS`: `267.34`
- `CPU FPS`: `1.40`

Zgodnie z przyjetym zalozeniem:

- architektura bez zapisanych wynikow w `results/` jest uznana za porazke,
- sukcesem jest tylko wariant, ktory poprawil sie wzgledem `FastSCNN`,
- w praktyce glownym kryterium sukcesu jest poprawa `mIoU`,
- dodatkowo zaznaczam, czy sukces byl pelny, tzn. czy poprawa jakosci nie odbyla sie kosztem GPU FPS.

## Wersje FAscnn_pp z wynikami

### Porazki wzgledem FastSCNN

- `FAscnn_pp_V3`: `mIoU 0.539405`, `GPU FPS 46.18`
- `FAscnn_pp_V13`: `mIoU 0.527228`, `GPU FPS 50.08`
- `FAscnn_pp_V14`: `mIoU 0.528398`, `GPU FPS 49.85`
- `FAscnn_pp_V15`: `mIoU 0.359985`, `GPU FPS 165.98`
- `FAscnn_pp_V16`: `mIoU 0.415146`, `GPU FPS 99.34`

### Sukcesy wzgledem FastSCNN

- `FAscnn_pp_V17`: `mIoU 0.643604`, `GPU FPS 300.56`
- `FAscnn_pp_V18`: `mIoU 0.654981`, `GPU FPS 296.70`

Interpretacja:

- `FAscnn_pp_V17` to pierwszy wyrazny sukces: lepsze `mIoU` niz `FastSCNN` i jednoczesnie lepszy `GPU FPS`.
- `FAscnn_pp_V18` to najlepszy wariant w aktualnych wynikach: najwyzsze `mIoU` i nadal bardzo wysoka wydajnosc.

## Wersje FAscnn_pp bez wynikow

Brak wynikow w `results/`:

- `FAscnn_pp_V6`
- `FAscnn_pp_V11`
- `FAscnn_pp_V12`

Zgodnie z przyjetym kryterium te wersje nalezy uznac za porazke eksperymentalna, bo nie ma materialu do porownania.

## Wniosek do pracy

Najwazniejszy wniosek jest taki, ze wiekszosc wariantow FAscnn_pp nie poprawila wyniku wzgledem `FastSCNN`.
Za realne sukcesy mozna uznac tylko `FAscnn_pp_V17` i `FAscnn_pp_V18`, przy czym najmocniejszym kandydatem
do opisu jako najlepsza architektura jest `FAscnn_pp_V18`. Poprawia ona `mIoU` bez utraty wydajnosci GPU.

## Na czym bazowaly kolejne architektury

### Faza 1: wczesne FAscnn_pp oparte glownie na ENet-like bottleneckach

- `FAscnn_pp_V3`: architektura encoder-decoder z trzema poziomami downsamplingu, bottleneckami ENet-like, `FastAttention` na kilku skalach i dekoderem opartym o `ConvTranspose2d`.
- `FAscnn_pp_V6`: bardziej rozbudowany wariant encoder-decoder oparty o bottlenecki typu `DDBottleNeck`, `ABottleNeck` i `UBottleNeck`.

### Faza 2: proby laczenia CNN z transformerem lub tokenami

- `FAscnn_pp_V11`: dwie galezie, z czego jedna byla prowadzona przez bottlenecki, a druga przez szybka uwage.
- `FAscnn_pp_V12`: iteracyjna architektura z `Multihead_Fast_Attention`, dekonwolucja i dwoma etapami fuzji.
- `FAscnn_pp_V13`: mocny reset w strone lekkiej architektury typu FastSCNN/BiSeNet. Pojawia sie wyrazny podzial na sciezke spatial i context, `PyramidPooling`, `FastAttention` na kontekscie oraz lekka fuzja.
- `FAscnn_pp_V14`: rozwinięcie `V13`, ale z tokenizacja mapy spatial przez `PatchEmbedConv`, blokiem token attention i odtworzeniem mapy przez `Unpatch`.
- `FAscnn_pp_V15`: jeszcze bardziej tokenowe podejscie. Zmniejszony stem, dwa kolejne bloki token-attention + FFN oraz nowa glowa `P2RefineHead`.
- `FAscnn_pp_V16`: kompromis po `V15`. Zostawiony tylko jeden blok tokenowy, dodana lepsza fuzja (`BetterFusionModule`), mocniejsza glowa `BetterP2RefineHead` oraz auxiliary i boundary heads.

### Faza 3: architektury wzorowane na FastSCNN/BiSeNet

- `FAscnn_pp_V17`: najczystsza i najbardziej udana baza. `LearningToDownsampleV2` jako spatial path, `GlobalFeatureExtractor` jako context path, `BiSeNetFFM` do fuzji i lekki `LiteClassifier`.
- `FAscnn_pp_V18`: `V17` plus `FastAttention` tylko na galezi context. To najlepszy wynik w calym repo.
