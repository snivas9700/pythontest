# History Log

The history log of the COPRA-WS.

# 1.0.3 (2017-08-14)

Bugfixes (ModelFactoryWS.py)

* For fixed-discounted products, start with Low, optimal=Med (Low +1), High (Low + 2), IntervalLow = IntervalHigh = optimal as Tomas suggested.

# 1.0.2 (2017-08-12)

Bugfixes (ModelFactoryWS.py)

* Added special handler for handling non_fixed_quotes-is-empty case (all items are fixed-dicounted products in the quote)

# 1.0.1 (2017-08-11)

Bugfixes (ModelFactoryWS.py)

* Fixed lab services fixed-discounted products functionality.
* Changed FixedDiscounting_Rules rule format.
* Added Country_SubReg_Geo_Mapping csv file.

## 1.0.0 (2017-08-08)

Improvements (ModelFactoryWS_08_08_17.py)

* Integration of Fixed-Discount Products (EMEA) and COPRA Engine. (see IntegrationDesign-FixedDiscountProducts&COPRAEngine.pdf)

## 0.0.0 (2017-05-17)

Get from iPAT team (ModelFactoryWS_24_05_17.py).