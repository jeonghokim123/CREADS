# Data sources

The input tables cover 584 regions in 26 countries; 340 regions have HDI labels. They are the processed inputs supplied with the final manuscript, not raw imagery or household-level survey records.

## Global Data Lab

The 2023 HDI labels were checked against **Subnational HDI Database v10.2** (released 13 May 2026). All 340 labelled values agree with that release. The source is the [Global Data Lab](https://globaldatalab.org/shdi/); the release is available from its [data archive](https://globaldatalab.org/shdi/archive/). This comparison was made on 8 November 2026.

GDL permits noncommercial use, adaptation and sharing with source and version attribution. Commercial use requires written permission. See the [GDL terms of use](https://globaldatalab.org/termsofuse/). The workbooks also retain additional GDL indicators and World Bank columns supplied with the analysis inputs; these are not all prediction variables. Their presence does not imply a common observation year. [World Bank data terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) apply to those indicators.

## Geospatial inputs

The processed geospatial columns are derived from VIIRS nighttime lights, GHSL population and settlement classes, ESA WorldCover, and administrative and accessibility data. The manuscript describes their preparation. Source products can be obtained from the [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/); provider terms continue to apply to their use.

Figure 1 downloads **GADM 4.1** boundary files from [GADM](https://gadm.org/). Boundary datasets are not included in this public release. GADM permits academic and other noncommercial use; redistribution requires prior permission, subject to the country-specific exceptions in its [licence](https://gadm.org/license.html).

The small-scale world basemap included with Figure 1 is from **Natural Earth**, whose map data are in the [public domain](https://www.naturalearthdata.com/about/terms-of-use/).

Code and derived research outputs do not confer additional rights over third-party data. This repository does not apply a blanket open-data licence to the input bundle.
