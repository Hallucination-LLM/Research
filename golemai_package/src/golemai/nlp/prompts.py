QUERY_INTRO = """Given the context `CONTEXT` and the query `QUERY` below, please provide an answer `ANSWER` to the question. 
    `CONTEXT`: {context} 

    `QUERY`: {query}

    `ANSWER`: {answer}
"""

QUERY_INTRO_NO_ANS = """Given the context `CONTEXT` and the query `QUERY` below, please provide an answer `ANSWER` to the question. 
    `CONTEXT`: {context} 

    `QUERY`: {query}

    `ANSWER`:
"""

QUERY_INTRO_FEWSHOT = """Given the context `CONTEXT` and the query `QUERY` below, please provide
  an answer `ANSWER` to the question. Three examples are provided below.

  `CONTEXT`: `Dokument[1]:` Zamykanie błędnych zleceń w SOZUW Zamknięcie błędnego zlecenia na MF (opisane
  w rozdziale 7) musi być poprzedzone zamknięciem tego zgłoszenia w SOZUW i przekazaniem
  informacji do autora zlecenia o przyczynach błędu i, jeśli jest to możliwe, sposobie
  obejścia problemu. Ten krok wykonujemy tylko po uzyskaniu takiej informacji od Mirka
  Cierluka/Rafała Markiewicza lub administratora SOZUW na czacie SOZUW - healthcheck
  na teams z wyjątkiem błędu sqlcode -302 występującego wraz z opisem Nie znaleziono
  NIP w CRP . SOZUW dostępny jest z poziomu przeglądarki na RDCB pod adresem https://sozuw.zus.ad.
  Po zalogowaniu się wybieramy menu Zamówienia . Następnie wyszukujemy interesujące
  nas zamówienie (zlecenie) – wpisujemy numer Zxxxxxxx i klikamy przycisk Szukaj .
  Wyświetlamy szczegóły interesującego nas zamówienia - przycisk Szczegóły . W widoku
  szczegółów zamówienia mamy dostępną informację o użytkowniku, który zlecił dane
  zadanie. Informacja ta będzie nam potrzebna do poinformowania danej osoby o przyczynie
  błędu i sposobie jego obejścia. W polu Nowy status ustawiamy status na Błędny a
  w polu Komentarz wpisujemy przyczynę ustawienia takiego statusu, następnie zatwierdzamy
  przyciskiem Zmień Status

  `QUERY`: Podaj mi link do sozuw

  `ANSWER`: W tekście wymieniono, że SOZUW dostępny jest z poziomu przeglądarki na RDCB
  pod adresem https://sozuw.zus.ad, dlatego link do sozuw to https://sozuw.zus.ad.


  `CONTEXT`: `Dokument Wskue.docx :` Opis procesu przetwarzania – MF: Uruchomienie WSKUE Uruchomienie WSKUE
  Po otrzymaniu maila o zakończeniu zasilania osoby realizujące przetwarzanie APER
  rozpoczynają przetwarzanie w OpenShift – po jego zakończeniu wysyłają maila na grupę
  PUS-E-ZCS_MONITOROWANIE z informacją o potrzebie uruchomienia procesu WSKUE (podusługa
  TWS: APS#01.WSK).

  Historia zmian Historia zmian Historia wersji dokumentu | Historia wersji dokumentu
  | Historia wersji dokumentu            | Historia wersji dokumentu Wersja                    |
  Data                      | Autor                                | Opis zmian 1.0                       |
  2020-07-21                | Jacek Stwora                         | Utworzenie dokumentu.
  1.1                       | 2020-07-23                | Witold Wilczyński Wojciech
  Lenkowski | Korekta dokumentu. 1.2                       | 2020-07-29                |
  Kamila Kamraj Wojciech Lenkowski     | Korekta i uzupełnienie dokumentu. 1.3                       |
  2020-07-29                | Rafał Markiewicz                     | Korekta dokumentu.
  1.4                       | 2020-09-15                | Mirosław Cierluk                     |
  Dodanie punktu 2.4


  `QUERY`: kto jest autorem tego dokumentu?

  `ANSWER`: Aby znaleźć autora dokumentu, należy sprawdzić kto jest autorem pierwszej
  wersji dokumentu. W tabeli zamieszczona jest informacja, że autorem wersji 1.0 jest
  Jacek Stwora, dlatego autorem tego dokumentu jest Jacek Stwora.


  `CONTEXT`: `Dokument MF-uprawnienia.docx :` Dodawanie uprawnień Po zalogowaniu do ZWAP wybierać opcję: R – RACF; zatwierdzić
  [ENTER]. Następnie opcję: 1 – DATA SET PROFILES; zatwierdzić [ENTER]. Dodać profil
  opcją: 1 – ADD; zatwierdzić [ENTER]. Wpisać w uszach nazwę datasetu z dwoma gwiazdkami,
  np.: ‘PKKAM.BMCSDEF.**’ i zatwierdzić [ENTER] ten oraz kolejny widok. Przejść do
  opcji: 4 – ACCESS i dla utworzonego profilu wybrać opcję 1 – ADD; zatwierdzić [ENTER].
  Zmienić SPECIFY z NO na YES; zatwierdzić [ENTER]. W polu AUTHORITY wpisać uprawnienie,
  np. READ, a polu users or groups wpisać nazwę użytkownika, np. PRMAR; zatwierdzić
  [ENTER]. Wybierając opcję D lub 8 można podejrzeć nadane uprawnienia.

  `QUERY`: Jak zamknąć zlecenie w SOZUW?

  `ANSWER`: Podany kontekst zawiera informacje o nadawaniu uprawnień. W żadnym miejscu
  we fragmencie nie znajduje się zmianka o SOZUW. Nie mogę udzielić odpowiedzi na
  to pytanie na podstawie podanego kontekstu.


  `CONTEXT`: {context}

  `QUERY`: {query}

  `ANSWER`:"""

SYSTEM_MSG_RAG_SHORT = """
    You are a helpful assistant. Your job will be to answer questions accurately based on the given context and not your internal knowledge.
    If you can not answer the question only based on the provided context, return the answer: `Nie mogę udzielić odpowiedzi na to pytanie na podstawie podanego kontekstu`.
"""

SYSTEM_MSG_SUM_SHORT = """
    You are a helpful assistant. Your job will be to answer questions accurately based on the given context and not your internal knowledge.
"""

SYSTEM_MSG_RAG = """
    You are a helpful assistant. Your job will be to answer questions accurately based on the given context and not your internal knowledge.
    If you can not answer the question only based on the provided context, return the answer: `Nie mogę udzielić odpowiedzi na to pytanie na podstawie podanego kontekstu`.
    Pay special attention to the names of applications, services, tools, and components - it is crucial to return consistent information for the subject. 
    Think of it step by step: 
        1. Find relevant information in the provided context. 
        2a. If there is no information relevant to the query, return the answer: `Nie mogę udzielić odpowiedzi na to pytanie na podstawie podanego kontekstu`
        2b. If information is relevant to the query, based on the context"s relevant information, formulate the final answer.
Your answers MUST be written in the language used in the question (can only be POLISH or ENGLISH).
The context will be provided by `CONTEXT`, the user query by `QUERY`, and your job is to return the answer `ANSWER`. `CONTEXT` is divided into several chunks which are introduced with the information in the format: `Dokument[ "{name}" ]:` or `Dokument[{number}]:`. For example: `Dokument[ "ProceduraMinisterstwo_v2.docx" ]:` or `Dokument[1]:`
If it is possible, use that information to infer the name of the document from which the context comes.
"""

SYSTEM_MSG_NER = """Your task is to perform named entitity recognition (NER) on the given text. 
You are supposed to return only valid JSON. 
Entities to extract will be provided in JSON schema. 
If it is not possible to extract such entity:  
    1. for `string` type entity return empty string: `"`     
    2. for `array` type entity return empty list: `[]`"""

placeholder = """
`TEXT`: "ZAŚWIADCZENIE NR YE/347740/1976/1544 O POMOCY DE MINIMIS

Data wydania
18-03-2014

A. CEL DOKUMENTU

[] Wydanie zaświadczenia [] Korekta zaświadczenia

Stwierdza się nieważność zaświadczenia nr  wydanego w dniu 

B. INFORMACJE DOTYCZĄCE PODMIOTU UDZIELAJĄCEGO POMOCY DE MINIMIS

Pieczęć

Numer identyfikacji podatkowej (NIP) podmiotu udzielającego pomocy de minimis
8255502800

Nazwa podmiotu udzielającego pomocy de minimis
Styn S.A.

Adres siedziby podmiotu udzielającego pomocy de minimis
ulica Mostowa 40/20, 49-349 Kłodzko

C1. INFORMACJE DOTYCZĄCE WNIOSKODAWCY
C. INFORMACJE DOTYCZĄCE BENEFICJENTA
NIEBĘDĄCEGO BENEFICJENTEM POMOCY DE
POMOCY DE MINIMIS
MINIMIS

Numer identyfikacji podatkowej (NIP) beneficjenta pomocy de minimis Numer identyfikacji podatkowej (NIP) wnioskodawcy
1971881179 

Imię i nazwisko albo nazwa beneficjenta pomocy de minimis Imię i nazwisko albo nazwa wnioskodawcy
Żabierek-Sobolak i syn s.c. 

Adres miejsca zamieszkania albo siedziby beneficjenta
pomocy de minimis
Adres miejsca zamieszkania albo siedziby wnioskodawcy
ul. Zachodnia 362, 31-661 Nowy Targ 

D. INFORMACJE DOTYCZĄCE UDZIELONEJ POMOCY DE MINIMIS

Poświadcza się, że pomoc udzielona w dniu 11-06-2021
na podstawie Kodeks rodzinny i opiekuńczy, art. 95, pkt. 3
o wartości brutto 900914 zł, stanowiącej równowartość 555,152 euro
stanowi pomoc de minimis.

Strona 1 z 2

Pomoc de minimis spełnia warunki określone w rozporządzeniu Komisji (należy zaznaczyć jedną z dwóch opcji):

[ ] (UE) NR 2023/2831 Z DNIA 13 GRUDNIA 2023 R. W SPRAWIE STOSOWANIA ART. 107 I 108 TRAKTATU O FUNKCJONOWANIU
UNII EUROPEJSKIEJ DO POMOCY DE MINIMIS (Dz. Urz. UE L 2023/2831 z 15.12.2023)

[ ] (UE) NR 2023/2832 Z DNIA 13 GRUDNIA 2023 R. W SPRAWIE STOSOWANIA ART. 107 I 108 TRAKTATU
O FUNKCJONOWANIU UNII EUROPEJSKIEJ DO POMOCY DE MINIMIS PRZYZNAWANEJ PRZEDSIĘBIORSTWOM
WYKONUJĄCYM USŁUGI ŚWIADCZONE W OGÓLNYM INTERESIE GOSPODARCZYM (Dz. Urz. UE L 2023/2832 z 15.12.2023)

Opis usługi świadczonej w ogólnym interesie gospodarczym
Światło odmiana dać.

E. DANE OSOBY UPOWAŻNIONEJ DO WYDANIA ZAŚWIADCZENIA

Imię i nazwisko Podpis
Melania Fiołka
M. Fiołka

Stanowisko służbowe
Sufler"

`JSON_SCHEMA`: {{"properties": {{
    "nip_udzielajacego_pomocy": {{"title": "Nip Udzielajacego Pomocy", "type": "string"}}, 
    "nazwa_udzielajacego_pomocy": {{"title": "Nazwa Udzielajacego Pomocy", "type": "string"}}, 
    "data_pomocy": {{"title": "Data Pomocy", "type": "string"}}, 
    "kwota_pomocy_pln": {{"title": "Kwota Pomocy Pln", "type": "string"}}, 
    "kwota_pomocy_euro": {{"title": "Kwota Pomocy Euro", "type": "string"}}}}, 
"required": ["nip_udzielajacego_pomocy", "nazwa_udzielajacego_pomocy", "data_pomocy", "kwota_pomocy_pln", "kwota_pomocy_euro"], 
"title": "AnswerFormat", "type": "object"}}
`OUTPUT`: {{"nip_udzielajacego_pomocy": "8255502800", "nazwa_udzielajacego_pomocy": "Styn S.A.", "data_pomocy": "11-06-2021", "kwota_pomocy_pln": "900914", "kwota_pomocy_euro": "555,152"}}"""


USER_MSG_NER = """Given the text `TEXT` and the json schema `JSON_SCHEMA` below, please provide
an output `OUTPUT` which is a valid JSON with extracted entities. One example is provided below:

`TEXT`: "ZAŚWIADCZENIE NR 1) O POMOCY DE MINIMIS
Data wydania
13-03-2006
A. CEL DOKUMENTU
Wydanie zaświadczenia Korekta zaświadczenia2)
Stwierdza się nieważność zaświadczenia nr 1) wydanego w dniu 15-09-2023
B. INFORMACJE DOTYCZĄCE PODMIOTU UDZIELAJĄCEGO POMOCY DE MINIMIS
Pieczęć
Numer identyfikacji podatkowej (NIP) podmiotu udzielającego pomocy de minimis
1952874103
Nazwa podmiotu udzielającego pomocy de minimis
Stowarzyszenie Piechuva-Burzec Sp. zo.o. Sp.k.
Adres siedziby podmiotu udzielającego pomocy de minimis
Ulica JabToniowa 81, 63-660 Radom
C1. INFORMACJE DOTYCZĄCE WNIOSKODAWCY
C. INFORMACJE DOTYCZĄCE BENEFICJENTA
NIEBĘDĄCEGO BENEFICJENTEM POMOCY DE
POMOCY DE MINIMIS3) MINIMIS4)
Numer identyfikacji podatkowej (NIP) beneficjenta pomocy de minimis Numer identyfikacji podatkowej (NIP) wnioskodawcy5)
5959816921 9790605468
Imię i nazwisko albo nazwa beneficjenta pomocy de minimis Imię i nazwisko albo nazwa wnioskodawcy
Olearczyk-Szulist isyn s.c. Maurycy Lewek
Adres miejsca zamieszkania albo siedziby beneficjenta
pomocy de minimis Adres miejsca zamieszkania albo siedziby wnioskodawcy
pl. Szorgtliwa 58, 38-072 Koszalin plac Norwida 66, 46-043 Lebork
D. INFORMACJE DOTYCZĄCE UDZIELONEJ POMOCY DE MINIMIS
Poświadcza się, że pomoc udzielona w dniu 20-12-2009
na podstawie6) Prawo o ruchu drogowym, art. 97, pkt. 4
o wartości brutto7) 27978 zł, stanowiącej równowartość 554,506 euro
stanowi pomoc de minimis.
Strona 1 z 2
Pomoc de minimis spełnia warunki określone w rozporządzeniu Komisji (należy zaznaczyć jedną z dwóch opcji):
(UE) NR 2023/2831 Z DNIA 13 GRUDNIA 2023 R. W SPRAWIE STOSOWANIA ART. 107 | 108 TRAKTATU O FUNKCJONOWANIU
UNII EUROPEJSKIEJ DO POMOCY DE MINIMIS (Dz. Urz. UE L 2023/2831 z 15.12.2023)
(UE) NR 2023/2832 Z DNIA 13 GRUDNIA 2023 R. W SPRAWIE STOSOWANIA ART. 107 | 108 TRAKTATU
O FUNKCJONOWANIU UNII EUROPEJSKIEJ DO POMOCY DE MINIMIS PRZYZNAWANEJ PRZEDSIĘBIORSTWOM
WYKONUJĄCYM USŁUGI ŚWIADCZONE W OGÓLNYM INTERESIE GOSPODARCZYM (Dz. Urz. UE L 2023/2832 z 15.12.2023)
Opis usługi świadczonej w ogólnym interesie gospodarczym8)
Zamiar szkoTa wartość padać pisać
E. DANE OSOBY UPOWAŻNIONEJ DO WYDANIA ZAŚWIADCZENIA
Imię i nazwisko Podpis
Maksymilian Mikus
Stanowisko służbowe Maksymilian Mikus
Instruktor"

`JSON_SCHEMA`: {{"properties": {{
    "nip_udzielajacego_pomocy": {{"title": "Nip Udzielajacego Pomocy", "type": "string"}}, 
    "nazwa_udzielajacego_pomocy": {{"title": "Nazwa Udzielajacego Pomocy", "type": "string"}}, 
    "data_pomocy": {{"title": "Data Pomocy", "type": "string"}}, 
    "kwota_pomocy_pln": {{"title": "Kwota Pomocy Pln", "type": "string"}}, 
    "kwota_pomocy_euro": {{"title": "Kwota Pomocy Euro", "type": "string"}}}}, 
"required": ["nip_udzielajacego_pomocy", "nazwa_udzielajacego_pomocy", "data_pomocy", "kwota_pomocy_pln", "kwota_pomocy_euro"], 
"title": "AnswerFormat", "type": "object"}}
`OUTPUT`: {{"nip_udzielajacego_pomocy": "1952874103", "nazwa_udzielajacego_pomocy": "Stowarzyszenie Piechuva-Burzec Sp. zo.o. Sp.k.", "data_pomocy": "20-12-2009", "kwota_pomocy_pln": "27978", "kwota_pomocy_euro": "554,506"}}

`TEXT`: {text}
`JSON_SCHEMA`: {json_schema}
`OUTPUT`: """



PROMPT_QA = (
    "You will be provided with a document and a proposed answer to a question. Your task is to determine if the proposed answer can be directly inferred from the document. "
    "If the answer contains any information not found in the document, it is considered false. Even if the answer is different from a ground truth answer, it might still be true, "
    "as long as it doesn't contain false information.\nFor each proposed answer, explain why it is true or false based on the information from the document. "
    "Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed answer "
    "is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact "
    "phrases or name entities from the answer that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the answer, in Python list of strings format].**"
    "\n\n#Document#: {document}\n\n#Ground Truth Answers#: {gt_response}\n\n#Proposed Answer#: {response}"
    "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, "
    "or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the answer, in a list of strings]** if your conclusion is 'False'."
)

PROMPT_SUMMARIZATION = (
    "You will be provided with a document and a proposed summary. Your task is to determine if the proposed summary can be directly inferred from the document. "
    "If the summary contains any information not found in the document, it is considered false. Even if the summary is different from a ground truth summary, "
    "it might still be true, as long as it doesn't contain false information.\nFor each proposed summary, explain why it is true or false based on the information from the document. "
    "Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed summary "
    "is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact "
    "phrases or name entities from the summary that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the summary, in Python list of strings format].**"
    "\n\n#Document#: {document}\n\n#Ground Truth Summary#: {gt_response}\n\n#Proposed Summary#: {response}"
    "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed summary is completely accurate based on the document, "
    "or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the summary, in a list of strings]** if your conclusion is 'False'."
)