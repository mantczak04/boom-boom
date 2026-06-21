def test_app_imports():
    from app import main

    assert callable(main)
