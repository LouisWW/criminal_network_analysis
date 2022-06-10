Explain src folder



---

## Profiling the code

To profile the code, the cprofile package was used together with the gprof package to visualise the critical part. To conduct such an analysis, please use the following commands

    $ python3 -m cProfile -o output.pstats main.py -read-data montagna_calls -sim-mart-vaq
    $ gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
    $ python3 -c"import pstats;p = pstats.Stats('output.pstats');p.sort_stats('cumulative').print_stats('simulators')"
