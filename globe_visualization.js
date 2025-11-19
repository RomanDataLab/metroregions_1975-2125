// Globe.gl Visualization Script
// Extracted from index.html

// Required imports
import { csvParse } from 'https://esm.sh/d3-dsv';
import { scaleSequentialSqrt } from 'https://esm.sh/d3-scale';
import { interpolateYlOrRd } from 'https://esm.sh/d3-scale-chromatic';

// Configuration - Generate years array from 1975 to 2125
const YEARS = Array.from({length: 151}, (_, i) => 1975 + i);
let allData = [];
let allDataByYear = {}; // Store data by year for quick access
let cityDataMap = new Map(); // Map to store city metadata by lat/lng
let allCitiesStable = []; // Stable list of all unique cities across all years
let pointsDataArray = []; // Persistent array of point objects for Globe.gl (updated in place)
let targetPopulations = new Map(); // Target population values for smooth interpolation
let currentRenderedPopulations = new Map(); // Track current rendered population values (inherited from previous year)
let isTransitioning = false; // Flag to track if we're transitioning
let currentYearIndex = 0;
let isPlaying = false;
let playInterval = null;

// Color scale for population visualization
const weightColor = scaleSequentialSqrt(interpolateYlOrRd)
  .domain([0, 5e7]); // Updated domain for larger populations

// Initialize Globe.gl visualization
const world = new Globe(document.getElementById('globeViz'))
  .globeImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/earth-night.jpg')
  .bumpImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png')
  .backgroundImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/night-sky.png')
      // City cylinders - height represents population
      .pointsData([]) // Will be populated with city data
      .pointId(d => d.cityKey) // Unique identifier for each city to enable smooth transitions
      .pointAltitude(d => d.pop * 5e-8) // Height proportional to population
      .pointRadius(0.4) // Base radius of cylinders
      .pointColor(d => weightColor(d.pop)) // Color based on population
      .pointOpacity(d => d.pop > 0 ? 1 : 0) // Make cities with pop=0 invisible
      .pointResolution(6) // Cylinder resolution (sides)
      .pointsTransitionDuration(1000)
      .pointsMergeTransitionDuration(1000) // Duration for merging/updating existing points
      .enablePointerInteraction(true)
  // Add country borders/contours
  .polygonsData([]) // Will be populated with country data
  .polygonAltitude(0.01) // Slight elevation for borders
  .polygonCapColor(() => 'rgba(100, 150, 200, 0.3)') // Light blue fill
  .polygonSideColor(() => 'rgba(100, 150, 200, 0.5)') // Border color
  .polygonStrokeColor(() => 'rgba(150, 200, 255, 0.8)') // Border outline
  .polygonsTransitionDuration(300);

// Add auto-rotation
world.controls().autoRotate = true;
world.controls().autoRotateSpeed = 0.6;

// Load country borders GeoJSON data
fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
  .then(res => res.json())
  .then(countries => {
    // Transform GeoJSON to format expected by Globe.gl
    const countryPolygons = countries.features.map(feature => ({
      geometry: feature.geometry,
      properties: feature.properties
    }));
    
    world.polygonsData(countryPolygons);
    console.log(`Loaded ${countryPolygons.length} country borders`);
  })
  .catch(err => {
    console.warn('Could not load country borders from primary source:', err);
    // Try alternative source - Natural Earth data
    fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')
      .then(res => res.json())
      .then(data => {
        const countryPolygons = data.features.map(feature => ({
          geometry: feature.geometry,
          properties: feature.properties
        }));
        world.polygonsData(countryPolygons);
        console.log(`Loaded ${countryPolygons.length} country borders from alternative source`);
      })
      .catch(err2 => {
        console.error('Failed to load country borders:', err2);
      });
  });

// Load metro_after.csv and transform data
fetch('metro_after.csv')
  .then(res => {
    if (!res.ok) {
      throw new Error('Failed to load metro_after.csv');
    }
    return res.text();
  })
  .then(csv => {
    const rows = csvParse(csv);
    
    // Transform data: extract lat/lng and all Pop_ columns
    allData = [];
    
    rows.forEach(row => {
      const lat = parseFloat(row.PWCent_Latitude);
      const lng = parseFloat(row.PWCent_Longitude);
      
      // Skip if lat/lng are invalid
      if (isNaN(lat) || isNaN(lng)) return;
      
      // Create a key for this city location
      const cityKey = `${lat.toFixed(4)}_${lng.toFixed(4)}`;
      
      // Store city metadata
      cityDataMap.set(cityKey, {
        cityName: row.City_Name || '',
        country: row.Location || '',
        areaKm2: parseFloat(row.AREA_km2_1Jan) || parseFloat(row.AREA_km2) || 0,
        cityCode: row.City_Code || ''
      });
      
      // Extract population for each year from Pop_ columns
      YEARS.forEach(year => {
        const popCol = `Pop_${year}`;
        const popValue = parseFloat(row[popCol]);
        
        // Skip if population is invalid or 0
        if (isNaN(popValue) || popValue <= 0) return;
        
        // Convert from thousands to actual population
        const pop = popValue * 1000;
        
        allData.push({
          lat: lat,
          lng: lng,
          year: year,
          pop: pop,
          cityName: row.City_Name || '',
          cityCode: row.City_Code || '',
          cityKey: cityKey
        });
        
        // Also store in a year-based structure for quick lookup
        if (!allDataByYear[year]) {
          allDataByYear[year] = [];
        }
        allDataByYear[year].push({
          lat: lat,
          lng: lng,
          year: year,
          pop: pop,
          cityName: row.City_Name || '',
          cityCode: row.City_Code || '',
          cityKey: cityKey
        });
      });
    });
    
    console.log(`Loaded ${rows.length} cities with data for ${YEARS.length} years`);
    console.log(`Total data points: ${allData.length}`);
    
    // Build stable list of all unique cities across all years
    // Use City_Code as primary key to prevent duplicates (more reliable than lat/lng)
    const cityCodesSet = new Set();
    const cityKeyByCode = new Map(); // Map City_Code to cityKey (lat_lng)
    
    rows.forEach(row => {
      const lat = parseFloat(row.PWCent_Latitude);
      const lng = parseFloat(row.PWCent_Longitude);
      const cityCode = row.City_Code || '';
      
      if (isNaN(lat) || isNaN(lng) || !cityCode) return;
      
      // Use City_Code as the unique identifier to prevent duplicates
      if (!cityCodesSet.has(cityCode)) {
        cityCodesSet.add(cityCode);
        const cityKey = `${lat.toFixed(4)}_${lng.toFixed(4)}`;
        cityKeyByCode.set(cityCode, cityKey);
        
        allCitiesStable.push({
          lat: lat,
          lng: lng,
          cityKey: cityKey,
          cityName: row.City_Name || '',
          cityCode: cityCode
        });
      }
    });
    
    // Sort stable cities by cityKey to ensure consistent ordering
    allCitiesStable.sort((a, b) => a.cityKey.localeCompare(b.cityKey));
    
    // Initialize persistent points data array with all cities
    // This array will be updated in place to maintain object references for smooth transitions
    const firstYear = YEARS[0];
    const firstYearData = allDataByYear[firstYear] || [];
    const firstYearMap = new Map();
    firstYearData.forEach(d => {
      firstYearMap.set(d.cityKey, d.pop);
    });
    
    pointsDataArray = allCitiesStable.map(city => {
      const initialPop = firstYearMap.get(city.cityKey) || 0;
      // Initialize current rendered populations map (for 1975, use 1975 data)
      currentRenderedPopulations.set(city.cityKey, initialPop);
      return {
        lat: city.lat,
        lng: city.lng,
        pop: initialPop,
        cityName: city.cityName,
        cityCode: city.cityCode,
        cityKey: city.cityKey,
        year: firstYear
      };
    });
    
    // Set initial points data - pass array directly to maintain reference
    world.pointsData(pointsDataArray);
    
    console.log(`Stable cities list: ${allCitiesStable.length} unique cities`);
    
    // Initialize UI with first year (1975) - data already set above
    currentYearIndex = 0;
    document.getElementById('yearDisplay').textContent = firstYear;
    document.getElementById('yearSlider').value = 0;
    
    // Update stats and rankings for first year
    const citiesWithPop = pointsDataArray.filter(d => d.pop > 0);
    const cityCount = citiesWithPop.length;
    const totalPop = citiesWithPop.reduce((sum, d) => sum + d.pop, 0);
    document.getElementById('cityCount').textContent = cityCount.toLocaleString();
    const formattedPop = totalPop >= 1e6 
      ? (totalPop / 1e6).toFixed(2) + 'M'
      : (totalPop / 1e3).toFixed(0) + 'K';
    document.getElementById('totalPop').textContent = formattedPop;
    updateRankings(firstYear);
  })
  .catch(err => {
    console.error('Error loading metro_after.csv:', err);
  });

// Update points data - inherit height from previous year's rendered state
function updatePointsData() {
  // CRITICAL: Update to target values
  // pointsDataArray already contains previous year's values (set in updateYear)
  // Now we update to target values - Globe.gl will animate from previous to target
  pointsDataArray.forEach(point => {
    const targetPop = targetPopulations.get(point.cityKey) || 0;
    point.pop = targetPop; // Set target - Globe.gl transitions from previous value to target
    
    // Update currentRenderedPopulations synchronously to the target value
    // This will be used as the starting point for the next year's transition
    // Even though the actual rendering happens asynchronously, our tracking is correct
    currentRenderedPopulations.set(point.cityKey, targetPop);
  });
  
  // Pass a shallow copy to trigger Globe.gl's change detection
  // Globe.gl uses pointId to match objects and will smoothly transition
  // from the previous value (already in point.pop before we updated) to the new target value
  world.pointsData([...pointsDataArray]);
}

// Update visualization for a specific year
function updateYear(yearIndex) {
  if (yearIndex < 0 || yearIndex >= YEARS.length) return;
  
  currentYearIndex = yearIndex;
  const year = YEARS[yearIndex];
  
  // Get data for current year - include ALL cities (even with 0 pop) for consistent tracking
  const currentYearData = allDataByYear[year] || allData
    .filter(d => d.year === year);
  
  // Create a map of target population values by cityKey
  // Include all cities to ensure Globe.gl can track them consistently
  targetPopulations.clear();
  currentYearData.forEach(d => {
    targetPopulations.set(d.cityKey, d.pop || 0);
  });
  
  // Ensure ALL cities in pointsDataArray have a target value (even if 0)
  // This ensures Globe.gl can track the same set of objects across all years
  pointsDataArray.forEach(point => {
    if (!targetPopulations.has(point.cityKey)) {
      targetPopulations.set(point.cityKey, 0);
    }
    point.year = year;
  });
  
  // When playing, directly set pointer heights to the population value for this year
  // No inheritance, just use the value from Pop_YEAR for the current year
  if (isPlaying) {
    // Directly set all pointers to the target population for this year
    pointsDataArray.forEach(point => {
      const yearPop = targetPopulations.get(point.cityKey) || 0;
      point.pop = yearPop;
      // Update tracking for consistency
      currentRenderedPopulations.set(point.cityKey, yearPop);
    });
    
    // Update Globe.gl directly with the year's population values
    world.pointsData([...pointsDataArray]);
    
    // Update UI immediately
    document.getElementById('yearDisplay').textContent = year;
    document.getElementById('yearSlider').value = yearIndex;
    
    // Update stats and rankings
    updateStatsAndRankings(year);
    
    // Set transition flag for play/pause logic
    isTransitioning = true;
    setTimeout(() => {
      isTransitioning = false;
    }, 1100); // Slightly longer than pointsTransitionDuration to ensure completion
    
    return; // Early return since we've handled everything for playing mode
  } else {
    // When not playing, use current rendered populations for smooth transitions
    pointsDataArray.forEach(point => {
      const previousRenderedPop = currentRenderedPopulations.get(point.cityKey);
      if (previousRenderedPop !== undefined) {
        point.pop = previousRenderedPop;
      } else {
        // Fallback: use current point.pop value
        const currentPop = point.pop || 0;
        point.pop = currentPop;
        currentRenderedPopulations.set(point.cityKey, currentPop);
      }
    });
    
    // Now update points data - Globe.gl will transition from current point.pop to target
    updatePointsData();
  }
  
  // Set transition flag for play/pause logic
  // Give Globe.gl time to complete the transition before allowing next update
  isTransitioning = true;
  setTimeout(() => {
    isTransitioning = false;
  }, 1100); // Slightly longer than pointsTransitionDuration to ensure completion
  
  // Update UI immediately
  document.getElementById('yearDisplay').textContent = year;
  document.getElementById('yearSlider').value = yearIndex;
  
  // Update stats and rankings with target values (will update during animation)
  updateStatsAndRankings(year);
}

// Update stats and rankings (called during animation)
function updateStatsAndRankings(year) {
  // Use current displayed values from pointsDataArray (will update smoothly during animation)
  const citiesWithPop = pointsDataArray.filter(d => d.pop > 0);
  const cityCount = citiesWithPop.length;
  const totalPop = citiesWithPop.reduce((sum, d) => sum + d.pop, 0);
  
  document.getElementById('cityCount').textContent = cityCount.toLocaleString();
  const formattedPop = totalPop >= 1e6 
    ? (totalPop / 1e6).toFixed(2) + 'M'
    : (totalPop / 1e3).toFixed(0) + 'K';
  document.getElementById('totalPop').textContent = formattedPop;
  
  // Update rankings (will use current animated values)
  updateRankings(year);
}

// Year slider event handler
document.getElementById('yearSlider').addEventListener('input', (e) => {
  if (!isPlaying) {
    updateYear(parseInt(e.target.value));
  }
});

// Play button event handler
document.getElementById('playBtn').addEventListener('click', () => {
  isPlaying = true;
  document.getElementById('playBtn').disabled = true;
  document.getElementById('pauseBtn').disabled = false;
  
  playInterval = setInterval(() => {
    if (!isPlaying) return;
    
    // Wait for current animation to complete before moving to next year
    if (isTransitioning) {
      return; // Skip this tick, will try again next interval
    }
    
    let nextIndex = currentYearIndex + 1;
    if (nextIndex >= YEARS.length) {
      nextIndex = 0; // Loop back to start
    }
    updateYear(nextIndex);
  }, 1500); // Check every 1500ms (animation takes 1000ms, give extra buffer)
});

// Pause button event handler
document.getElementById('pauseBtn').addEventListener('click', () => {
  isPlaying = false;
  document.getElementById('playBtn').disabled = false;
  document.getElementById('pauseBtn').disabled = true;
  
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
  }
  
  isTransitioning = false;
});

// Reset button event handler
document.getElementById('resetBtn').addEventListener('click', () => {
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
  }
  isPlaying = false;
  document.getElementById('playBtn').disabled = false;
  document.getElementById('pauseBtn').disabled = true;
  
  isTransitioning = false;
  
  updateYear(0);
});

// Rotation toggle button event handler
let rotationEnabled = true;
document.getElementById('rotationBtn').addEventListener('click', () => {
  rotationEnabled = !rotationEnabled;
  world.controls().autoRotate = rotationEnabled;
  
  const rotationBtn = document.getElementById('rotationBtn');
  if (rotationEnabled) {
    rotationBtn.textContent = 'Rotation: On';
  } else {
    rotationBtn.textContent = 'Rotation: Off';
  }
});

// Toggle ranking panel minimize/expand
window.toggleRanking = function(panelId) {
  const panel = document.getElementById(panelId);
  panel.classList.toggle('minimized');
}

// Format population for display
function formatPopulationDisplay(pop) {
  if (pop >= 1e6) {
    return (pop / 1e6).toFixed(2) + 'M';
  } else if (pop >= 1e3) {
    return (pop / 1e3).toFixed(1) + 'K';
  }
  return pop.toLocaleString();
}

// Update country rankings (Top 10 countries with metropolises over 1M)
function updateCountryRankings(year) {
  // Use pointsDataArray to ensure consistency with displayed data
  const yearData = pointsDataArray.filter(d => d.year === year && d.pop > 0);
  
  // Filter cities with population over 1M
  const citiesOver1M = yearData.filter(d => d.pop >= 1000000);
  
  // Group by country and count cities
  const countryCounts = new Map();
  citiesOver1M.forEach(city => {
    const cityInfo = cityDataMap.get(city.cityKey) || {};
    const country = cityInfo.country || 'Unknown';
    
    if (!countryCounts.has(country)) {
      countryCounts.set(country, 0);
    }
    countryCounts.set(country, countryCounts.get(country) + 1);
  });
  
  // Convert to array and sort by count (descending)
  const countryRankings = Array.from(countryCounts.entries())
    .map(([country, count]) => ({ country, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10); // Top 10
  
  // Update title
  document.getElementById('countriesTitle').textContent = 
    `Top 10 Countries with Metropolises over 1 mln by ${year}`;
  
  // Update table
  const tbody = document.getElementById('countriesRankingBody');
  tbody.innerHTML = countryRankings.map((item, index) => `
    <tr>
      <td class="rank-number">${index + 1}</td>
      <td class="country-name">${item.country}</td>
      <td class="cities-count">${item.count}</td>
    </tr>
  `).join('');
}

// Update city rankings (Top 10 metropolises by population)
function updateCityRankings(year) {
  // Use pointsDataArray to ensure consistency with displayed data
  const yearData = pointsDataArray.filter(d => d.year === year && d.pop > 0);
  
  // Sort by population (descending) and take top 10
  const cityRankings = yearData
    .map(city => ({
      cityName: city.cityName || 'Unknown',
      population: city.pop
    }))
    .sort((a, b) => b.population - a.population)
    .slice(0, 10); // Top 10
  
  // Update title
  document.getElementById('citiesTitle').textContent = 
    `Top 10 Metropolises by Population by ${year}`;
  
  // Update table
  const tbody = document.getElementById('citiesRankingBody');
  tbody.innerHTML = cityRankings.map((item, index) => `
    <tr>
      <td class="rank-number">${index + 1}</td>
      <td class="city-name">${item.cityName}</td>
      <td class="population-value">${formatPopulationDisplay(item.population)}</td>
    </tr>
  `).join('');
}

// Update all rankings for a given year
function updateRankings(year) {
  updateCountryRankings(year);
  updateCityRankings(year);
}

