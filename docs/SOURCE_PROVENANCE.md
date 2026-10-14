# Source provenance

I credit Dr. Tony Fisher's modem work and Randy Rossi's later rewrite in
[CREDITS](../CREDITS.md). Fisher's cached original readme requests retaining his
name and readme. It does not contain an explicit MIT grant. Rossi's repository
has a separate MIT notice naming Rossi; I do not treat that notice as proof of
permissions for every historical Fisher file.

Earlier El-Modem V.32 code was developed with Fisher's implementation in view.
The transmit scrambler, differential coders, 7200-bit/s trellis encoder and
tables were subsequently re-expressed from ITU-T V.32/V.32bis. The receiver's
differential decoding table is the inverse of the standard's encoding table.
The code no longer claims that this history was independent or clean-room.

The table audit derives seven sets of values from V.32bis Tables 1, 2 and 5 and
the convolutional/constellation figures. Regression tests and exhaustive
branch comparisons establish output equivalence to the earlier implementation.
These are technical conformance checks, not a determination of source permissions.

The receiver retains reference-informed algorithm choices, including a
20-symbol traceback and 0.99 survivor-metric decay. Those choices are recorded
in the source. Changing their numerical values would change tested behavior
without establishing independent authorship, so I have retained them.

The original Fisher source files and the proprietary SmartLink modem binary are
not included in this public project or its executable. SmartLink serves as an
independent interoperability peer; it is not part of El-Modem's implementation.
Third-party recordings and ITU PDFs are also excluded from distribution.

The community license for original El-Modem work is GPL-3.0-or-later. Dependency
licenses remain their own. This record preserves attribution and the remaining
review boundary; it does not create a license grant for somebody else's work.
See [redistribution review](REDISTRIBUTION.md) for binary dependencies and source
delivery, and [contribution policy](../CONTRIBUTING.md) before submitting code.
