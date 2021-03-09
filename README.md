# polarisation_tests_for_FEE
A notebook and some accompanying code for checking the FEE code plays nice
with Stokes Vectors. Everything is in `python`, and I'm using [hyperbeam](https://pypi.org/project/mwa-hyperbeam/) to calculate the FEE beam. Notebook contains all the theory (note the LaTeX renders fine on my desktop but not on github - I've included `polarised_source_and_FEE_beam.pdf` in case it doesn't render for you either), and contains the
following checks:

 - Simulating a source with linear polarisation, and checking that the FEE beam will observe the correct rotation measure
 - Rotating the FEE beam into instrumental polarisations, then observing a sky with a given Stokes parameter, will return the correct Stokes sky
 - The azimuth values we input mean the beam points to the correct RA/Dec
 - Comparing things to the RTS analytic beam as a sanity check

Any comments or concerns feel free to Slack me or email jack.line@curtin.edu.au. Also feel free to apply these tests to your implementation of the FEE beam.
