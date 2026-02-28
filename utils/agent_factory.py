def get_agent_class(algo_name):
    algo_name = algo_name.upper()
    if algo_name == 'MAIC':
        from mac import create_maic
        return create_maic
    elif algo_name == 'CMASAC':
        from mac import create_cmasac
        return create_cmasac
    elif algo_name == 'COMA':
        from mac import create_coma
        return create_coma
    elif algo_name == 'ISAC':
        from mac import create_isac
        return create_isac
    elif algo_name == 'MASAC':
        from mac import create_masac
        return create_masac
    else:
            raise ValueError(f"Unknown algorithm selected: {algo_name}")
