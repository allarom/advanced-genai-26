# Step 4 feedback set - MATCHED to benchmark type mix

10 faithful, corpus-grounded Q&A. Question types are engineered to mirror the 25-question benchmark distribution ({'entity_temporal': 3, 'entity': 4, 'keyword': 6, 'semantic': 8, 'mixed': 4}). Every answer is verified against its source ETH-News document.

| id | query_type | question | answer | source (doc_id, year) |
|----|------------|----------|--------|-----------------------|
| m01 | entity_temporal | Who became a professor of information technology and education at ETH Zurich in 2004? | Juraj Hromkovic | `751b434a93…`, 2017 |
| m02 | entity | Who is the director of the Zurich Information Security Center at ETH Zurich? | Srdjan Capkun | `602257ce87…`, 2017 |
| m03 | entity | Who is the professor of bioethics at ETH Zurich? | Effy Vayena | `04090bd666…`, 2020 |
| m04 | keyword | What is GratXray? | An ETH spin-off in X-ray imaging co-founded by Marco Stampanoni. | `37b43409ad…`, 2017 |
| m05 | keyword | Which prize did the ETH research team receive for their nanoelectronics simulation? | The ACM Gordon Bell Prize. | `1438012fde…`, 2019 |
| m06 | semantic | What research field does professor Markus Reiher work in at ETH Zurich? | Theoretical chemistry. | `4d1c201f42…`, 2017 |
| m07 | semantic | In which research area is Nenad Ban a professor at ETH Zurich? | Structural molecular biology. | `37a7c02b6b…`, 2016 |
| m08 | semantic | What kind of bioplastic does Massimo Morbidelli's group investigate at ETH Zurich? | Polyethylene furanoate (PEF). | `2746df51b3…`, 2018 |
| m09 | mixed | ETH professor of theoretical computer science? | Peter Widmayer | `7ed1ed33f7…`, 2020 |
| m10 | mixed | Control theory professor at ETH? | Mustafa Khammash | `09ac943888…`, 2016 |
