from .cli.run import main

if __name__ == '__main__':
    import sys
    from . import __name__ as module
    # `python -m <module>` typically displays the command as __main__.py
    if '__main__.py' in sys.argv[0]:
        sys.argv[0] = '%s -m %s' % (sys.executable, module)
    main()
