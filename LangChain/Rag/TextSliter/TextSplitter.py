

#importing module for spliting the text into chunks 
#
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
A swirling accretion disk around a supermassive black hole in the heart of a barred spiral galaxy, with rogue planets, neutron stars, and glowing nebulae orbiting chaotically
Stellar black hole devouring a red giant star inside an irregular galaxy cluster, surrounded by pulsar beams, dark matter halos, and cosmic dust clouds
Primordial black hole warping spacetime in a dwarf galaxy, pulling in asteroid fields, white dwarfs, and quantum foam particles
Intermediate-mass black hole at the core of a merging galaxy pair, with tidal streams of stars, quasar jets, and exotic matter streams
Spinning Kerr black hole in a lenticular galaxy, ejecting relativistic jets through rings of plasma, exoplanets, and gamma-ray bursts
Charged Reissner-Nordström black hole embedded in an elliptical galaxy, encircled by magnetars, brown dwarfs, and wormhole echoes
Supermassive black hole Sagittarius A* analog in a Milky Way-like galaxy, with orbiting S-stars, molecular clouds, and Hawking radiation flares
Micro black hole evaporating near a star-forming region in a spiral arm galaxy, surrounded by protostars, comets, and gravitational waves
Black hole binary system colliding in a galaxy collision zone, creating ripples of spacetime, hypervelocity stars, and supernova remnants
Event horizon of a black hole in a Seyfert galaxy, absorbing entire star clusters, interstellar gas, and rogue black holes
Singularity inside a black hole at the center of an active galactic nucleus, with surrounding relativistic jets, radio lobes, and cosmic filaments
Black hole feeding frenzy in a galaxy group, consuming globular clusters, planetary nebulae, and dark energy voids
Wormhole-black hole hybrid portal in a void galaxy, linking to parallel universes filled with alien megastructures and exotic particles
Hawking radiation glow from a tiny black hole in a young galaxy, illuminating protoplanetary disks, quasars, and magnetized plasma clouds
Accretion disk instability around a black hole in an ultra-luminous infrared galaxy, spawning new stars, black hole seeds, and tidal disruption events
Primordial black hole swarm drifting through a dwarf irregular galaxy, interacting with cosmic strings, gravitons, and shadow galaxies
Supermassive black hole merger in a cluster galaxy, releasing gravitational waves that distort nearby pulsars, neutron stars, and dark matter filaments
Isolated stellar black hole in the outskirts of a spiral galaxy, orbited by frozen rogue worlds, asteroid belts, and faint radio echoes
Charged black hole in a quasar host galaxy, emitting X-ray flares, gamma bursts, and streams of virtual particles
Black hole with an ergosphere in a rotating galaxy core, flinging plasma blobs, hypervelocity stars, and exotic matter
Dormant black hole awakening in an ancient elliptical galaxy, surrounded by fossil stars, globular clusters, and relic radiation
Black hole encircled by a Dyson swarm in a Kardashev Type II galaxy, with megastructures, AI probes, and quantum-entangled signals
Intermediate black hole in a starburst galaxy, triggering chain supernovae, molecular outflows, and new black hole formations
Black hole at the edge of a cosmic web filament, linked to galaxy filaments, voids, and large-scale structure threads
Rotating black hole with frame-dragging effects in a barred galaxy, twisting spacetime around magnetized neutron stars and blue supergiants
Evaporating black hole remnant in a high-redshift galaxy, scattering photons, gravitons, and primordial soup elements
Binary black hole system in a compact galaxy merger, with chaotic orbits, gravitational lensing, and ejected stars
Supermassive black hole feedback loop in a cool-core galaxy cluster, regulating star formation with jets, bubbles, and hot gas halos
Quantum black hole tunneling through a galaxy halo, creating mini-wormholes, particle showers, and spacetime foam bubbles
Ancient black hole relic from the Big Bang in a fossil galaxy, surrounded by cosmic microwave background echoes, dark energy, and void structures
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1, 
    chunk_overlap=0 
)

chunks = splitter.split_text(text);
print(chunks)
print("Number of chunks: ", len(chunks))                                                                                                                                                                                                                                                        