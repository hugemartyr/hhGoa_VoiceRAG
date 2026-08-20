"""Generate a synthetic MSMARCO dataset for pipeline testing.

HuggingFace streaming can be slow for testing. This script creates
a small list of dictionaries that mimics the structure of MSMARCO-XI.
"""

from typing import List, Dict, Any

def get_mock_dataset() -> List[Dict[str, Any]]:
    return [
        {
            "query_id": "1",
            "English_passages": [
                "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
                "This chemical energy is later released to fuel the organisms' activities.",
                "Cellular respiration is the process of converting glucose into usable energy in the form of ATP."
            ],
            "is_selected": [1, 0, 0],
            "query_type": "description"
        },
        {
            "query_id": "2",
            "English_passages": [
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                "It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
            ],
            "is_selected": [1, 1],
            "query_type": "entity"
        },
        {
            "query_id": "3",
            "English_passages": [
                # Long passage to trigger chunker splitting
                " ".join([
                    "Water (chemical formula H2O) is an inorganic, transparent, tasteless, odorless, and nearly colorless chemical substance.",
                    "It is the main constituent of Earth's hydrosphere and the fluids of all known living organisms.",
                    "It is vital for all known forms of life, despite providing neither food, energy, nor organic micronutrients.",
                    "Its chemical formula, H2O, indicates that each of its molecules contains one oxygen and two hydrogen atoms, connected by covalent bonds.",
                    "The hydrogen atoms are attached to the oxygen atom at an angle of 104.45 degrees.",
                    "Water is the name of the liquid state of H2O at standard temperature and pressure.",
                    "It forms precipitation in the form of rain and aerosols in the form of fog.",
                    "Clouds consist of suspended droplets of water and ice, its solid state.",
                    "When finely divided, crystalline ice may precipitate in the form of snow.",
                    "The gaseous state of water is steam or water vapor.",
                    "Water moves continually through the water cycle of evaporation, transpiration (evapotranspiration), condensation, precipitation, and runoff, usually reaching the sea.",
                    "Water plays an important role in the world economy.",
                    "Approximately 70% of the freshwater used by humans goes to agriculture.",
                    "Fishing in salt and fresh water bodies is a major source of food for many parts of the world.",
                    "Much of the long-distance trade of commodities is transported by boats through seas, rivers, lakes, and canals.",
                    "Large quantities of water, ice, and steam are used for cooling and heating, in industry and homes.",
                    "Water is an excellent solvent for a wide variety of substances both mineral and organic; as such it is widely used in industrial processes, and in cooking and washing."
                ])
            ],
            "is_selected": [1],
            "query_type": "description"
        }
    ]
