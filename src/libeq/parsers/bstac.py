from typing import Any


def parse_titration(lines, jw, icd, nc) -> list[dict[str, Any]]:
    """Parse the titration data section of a BSTAC file.

    Reads one or more titration blocks from *lines*, each containing the
    titration name, component settings, concentration data, background
    parameters, volume parameters, electrode parameters, and the individual
    volume/potential data points.

    Parameters
    ----------
    lines : list of str
        The raw text lines of the titration section (from the BSTAC file).
    jw : int
        Weighting mode flag from the BSTAC header (``MODE`` field).  When
        ``jw == 2``, a sigma column is expected for each data point.
    icd : int
        Ionic-strength-correction mode flag from the BSTAC header (``ICD``
        field).
    nc : int
        Number of chemical components (from the BSTAC header ``NC`` field).

    Returns
    -------
    list of dict
        One dictionary per titration block, containing keys such as
        ``"titration_name"``, ``"components_concentrations"``,
        ``"v_params"``, ``"potential_params"``, ``"volume"``,
        ``"potential"``, and ``"ignored"``.
    """
    tot_length = len(lines)
    sections = [
        (lambda line: line.strip(), 1, "titration_name"),  # NAMET
        (
            lambda line: [int(part) for part in line.split()],
            1,
            "titration_comp_settings",
        ),  # JP,NCET
        (
            lambda line: [
                float(part) if i != 2 else int(part)
                for i, part in enumerate(line.split())
            ],
            "NC",
            "components_concentrations",
        ),  # CO,CTT,LOK
        (
            lambda line: [float(part) for _, part in enumerate(line.split())],
            1,
            "background_params",
        ),  # COI,CTI,IREFT
        (
            lambda line: [float(part) for _, part in enumerate(line.split())],
            1,
            "v_params",
        ),  # VO,SIGMAV
        (
            lambda line: [
                float(part) if i < 5 else int(part)
                for i, part in enumerate(line.split())
            ],
            1,
            "potential_params",
        ),  # E0,SIGMAE,JA,JB,SLOPE,LOK1,LOK2,LOK3,LOK4
        (
            lambda line: [
                float(part) if i != 3 else int(part)
                for i, part in enumerate(
                    map(lambda x: x.replace("(", "").replace(")", ""), line.split())
                )
            ],
            "until_end",
            "titration_values",
        ),  # V,E,(SIGMA),IND
    ]
    line_counter = 0
    titrations = []
    while True:
        titration = {}
        for process_func, repeat, name in sections:
            if isinstance(repeat, int):
                for _ in range(repeat):
                    titration[name] = process_func(lines[line_counter])
                    line_counter += 1
            elif repeat == "NC":
                for _ in range(nc):
                    parsed_line = process_func(lines[line_counter])
                    parsed_line = {
                        k: v
                        for k, v in zip(
                            ["C0", "CTT", "LOK"],
                            parsed_line,
                        )
                    }
                    titration.setdefault(name, []).append(parsed_line)
                    line_counter += 1

            elif repeat == "until_end":
                while True:
                    parsed_line = process_func(lines[line_counter])
                    titration.setdefault("volume", []).append(parsed_line[0])
                    titration.setdefault("potential", []).append(parsed_line[1])
                    if jw == 2:
                        titration.setdefault("sigma", []).append(parsed_line[2])
                    titration.setdefault("ignored", []).append(parsed_line[-1] == -1)
                    line_counter += 1
                    if parsed_line[-1] == 1:
                        break

        titrations.append(titration)
        if line_counter == tot_length:
            break

    return titrations


def parse_model(lines, icd, nc) -> list[dict[str, Any]]:
    """Parse the species model section of a BSTAC file.

    Reads one line per species from *lines*, extracting the log stability
    constant (``BLOG``), stoichiometric coefficients (``IX1`` … ``IXnc``),
    refinement flag (``KEY``), and, when ``icd > 0``, the ionic-strength
    correction coefficients ``IB``, ``C``, ``D``, ``E``.

    Parameters
    ----------
    lines : list of str
        One text line per species from the BSTAC species block.
    icd : int
        Ionic-strength-correction mode (``ICD`` field from the BSTAC header).
        ``0`` means no correction; non-zero enables additional columns.
    nc : int
        Number of chemical components (``NC`` field from the BSTAC header).

    Returns
    -------
    list of dict
        One dictionary per species.  Common keys include ``"BLOG"``,
        ``"IX1"`` … ``"IXnc"``, ``"KEY"``, and ``"NKA"``.  When
        ``icd > 0``, ``"IB"``, ``"C"``, ``"D"``, ``"E"`` are also present.
    """
    species = []
    sections = [
        lambda line: [
            int(part) if i > 2 else float(part)
            for i, part in enumerate(
                map(lambda x: x.replace("(", "").replace(")", ""), line.split())
            )
        ],  # BLOG,IX(NC times),KEY,NKA,IKA(NKA times) (ICD=0)
        lambda line: [
            int(part) if i > 4 else float(part)
            for i, part in enumerate(
                map(lambda x: x.replace("(", "").replace(")", ""), line.split())
            )
        ],
        # BLOG,(IB),C,D,E,IX(1...NC),KEY,KEYC,KEYD,KEYE,NKA,IKA(1...NKA) (ICD=1/2)
    ]

    if icd == 0:
        process_func = sections[0]
        model_columns = (
            [
                "BLOG",
            ]
            + [f"IX{i}" for i in range(1, nc + 1)]
            + [
                "KEY",
                "NKA",
            ]
            + [f"IKA{i}" for i in range(1, 10)]
        )
    else:
        process_func = sections[1]
        model_columns = (
            [
                "BLOG",
                "IB",
                "C",
                "D",
                "E",
            ]
            + [f"IX{i}" for i in range(1, nc + 1)]
            + [
                "KEY",
                "KEYC",
                "KEYD",
                "KEYE",
                "NKA",
            ]
            + [f"IKA{i}" for i in range(1, 10)]
        )

    for line in lines:
        parsed_line = process_func(line)
        parsed_line = {
            k: v
            for k, v in zip(
                model_columns,
                parsed_line,
            )
        }
        species.append(parsed_line)

    return species


def parse_BSTAC_file(lines):
    """Parse the contents of a BSTAC-format input file.

    Reads successive sections of the BSTAC file (header control integers,
    component names, temperature, ionic-strength correction parameters,
    component charges, species model, and titration data) and assembles
    them into a single flat dictionary.

    Parameters
    ----------
    lines : list of str
        All lines of the BSTAC file as returned by :func:`open`.readlines`.

    Returns
    -------
    dict
        Flat dictionary with keys corresponding to the BSTAC fields, e.g.
        ``"MAXIT"``, ``"NC"``, ``"NS"``, ``"MODE"``, ``"ICD"``,
        ``"comp_name"``, ``"TEMP"``, ``"charges"``, ``"species"``, and
        ``"titrations"``.  Ionic-strength parameters (``"AT"``, ``"BT"``,
        ``"c0"`` … ``"e1"``) are included only when ``ICD > 0``.
    """
    # Define the list of tuples
    sections = [
        (lambda line: line.strip(), 1, "file_name"),  # TITLE
        (
            lambda line: [int(part) for part in line.split()],
            1,
            ["MAXIT", "NC", "NS", "MODE", "ICD", "WESP", "SHLIM"],
        ),  # MAXIT,NC,NS,MODE,ICD,WESP,SHLIM
        (lambda line: line.strip(), "NC", "comp_name"),  # COMP
        (
            lambda line: [float(part) for part in line.split()],
            1,
            ["TEMP", "PHI", "PHF"],
        ),  # TEMP,PHI,PHF
        (
            lambda line: [
                float(part) if i < 9 else int(part)
                for i, part in enumerate(line.split())
            ],
            "ICD",
            ["IREF", "AT", "BT", "c0", "c1", "d0", "d1", "e0", "e1", "KCD"],
        ),  # IREF,AT,BT,c0,c1,d0,d1,e0,e1,KCD(1...6)
        (
            lambda line: [int(part) for part in line.split()],
            1,
            "charges",
        ),  # Z(1...NC)
        (
            parse_model,
            "NS",
            "species",
        ),  # BLOG,(IB),C,D,E,IX(1...NC),KEY,KEYC,KEYD,KEYE,NKA,IKA(1...NKA)
        (parse_titration, "end_of_file", "titrations"),
    ]

    result = {}
    line_counter = 0

    for process_func, repeat, name in sections:
        if isinstance(repeat, int):
            for _ in range(repeat):
                if isinstance(name, str):
                    result[name] = process_func(lines[line_counter])
                elif isinstance(name, list):
                    for field_name, data in zip(
                        name, process_func(lines[line_counter])
                    ):
                        result[field_name] = data
                line_counter += 1
        elif repeat == "NC":
            nc = result["NC"]  # Get the value of NC from the data
            for _ in range(nc):
                result.setdefault(name, []).append(process_func(lines[line_counter]))
                line_counter += 1
        elif repeat == "ICD":
            icd = result["ICD"]  # Get the value of ICD from the data
            if icd > 0:
                for field_name, data in zip(name, process_func(lines[line_counter])):
                    result[field_name] = data
                line_counter += 1
            else:
                for field_name, data in zip(name, process_func(lines[line_counter])):
                    result[field_name] = None
        elif repeat == "NS":
            ns = result["NS"]  # Get the value of NS from the data
            parsed_section = process_func(
                lines[line_counter : line_counter + ns], icd, nc
            )
            result[name] = parsed_section
            line_counter += ns
        elif repeat == "end_of_file":
            parsed_section = process_func(lines[line_counter:], result["MODE"], icd, nc)
            result[name] = parsed_section

    return result
