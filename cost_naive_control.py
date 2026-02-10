from benchmark_cost import main


if __name__ == "__main__":
    import sys

    sys.argv.extend(["--mode", "naive"])
    main()
