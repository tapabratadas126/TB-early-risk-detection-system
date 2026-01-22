import React, { useState, useMemo } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { MapPin, Phone, ExternalLink, AlertTriangle, Search, Navigation as NavigationIcon, Globe } from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { getComprehensiveTranslation } from '../translations/comprehensive-translations';
import { allIndianStates, getDistrictsByStateName, searchStatesByPrefix, searchDistrictsByPrefix } from '../data/comprehensive-indian-locations';
import { expandedHospitalDatabase, getHospitalsByStateDistrict, type Hospital } from '../data/expanded-hospital-database';
import { 
  comprehensiveHospitalDatabase, 
  getAllUniqueCities, 
  searchCitiesByPrefix, 
  getHospitalsWithinRadius,
  type ComprehensiveHospital 
} from '../data/comprehensive-hospital-database';
import {
  westBengalDistricts,
  getAllWBCities,
  searchWBCityByPrefix,
  getWBPostOfficesByDistrict,
  type WBPostOffice
} from '../data/west-bengal-locations';
import {
  westBengalHospitals,
  getWBHospitalsByDistrict,
  getWBHospitalsByCity,
  getWBHospitalsByPinCode,
  getWBHospitalsByPostOffice,
  searchWBHospitals,
  type WBHospital
} from '../data/west-bengal-hospitals';
import {
  getCitiesByDistrict,
  getCitiesByState,
  districtCityData
} from '../data/district-city-mapping';

export default function FindHelpPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedState, setSelectedState] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [searchResults, setSearchResults] = useState<(Hospital | ComprehensiveHospital | WBHospital)[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [showNoResults, setShowNoResults] = useState(false);
  const { language } = useLanguage();
  const t = getTranslation(language);
  const comprehensiveT = getComprehensiveTranslation(language);

  // Filter states based on search query (for typing mode)
  const filteredStates = useMemo(() => {
    if (!searchQuery.trim()) return allIndianStates.map(s => s.name);
    return searchStatesByPrefix(searchQuery);
  }, [searchQuery]);

  // Filter districts based on selected state and search query
  const filteredDistricts = useMemo(() => {
    if (!selectedState) return [];
    const districts = getDistrictsByStateName(selectedState);
    if (!searchQuery.trim()) return districts;
    return districts.filter(d => d.toLowerCase().startsWith(searchQuery.toLowerCase()));
  }, [selectedState, searchQuery]);

  // Filter cities based on SELECTED DISTRICT - This is the key change!
  const filteredCities = useMemo(() => {
    // If district is selected, show only cities from that district
    if (selectedDistrict && selectedState) {
      const citiesInDistrict = getCitiesByDistrict(selectedDistrict, selectedState);
      if (!searchQuery.trim()) return citiesInDistrict;
      return citiesInDistrict.filter(city => 
        city.toLowerCase().startsWith(searchQuery.toLowerCase())
      );
    }
    
    // If only state is selected, show all cities in that state
    if (selectedState) {
      const citiesInState = getCitiesByState(selectedState);
      if (!searchQuery.trim()) return citiesInState;
      return citiesInState.filter(city => 
        city.toLowerCase().startsWith(searchQuery.toLowerCase())
      );
    }
    
    // No state/district selected - show all cities
    if (!searchQuery.trim()) return getAllUniqueCities();
    return searchCitiesByPrefix(searchQuery);
  }, [selectedDistrict, selectedState, searchQuery]);

  const handleSearch = () => {
    // Validate that at least state and district are selected
    if (!selectedState || !selectedDistrict) {
      setShowNoResults(true);
      setHasSearched(false);
      return;
    }

    setIsSearching(true);
    setHasSearched(false);
    setShowNoResults(false);
    
    // Simulate API call with delay
    setTimeout(() => {
      let results: (Hospital | ComprehensiveHospital | WBHospital)[] = [];

      // First priority: Check West Bengal specific database for more accurate results
      if (selectedState === 'West Bengal') {
        const wbResults = getWBHospitalsByDistrict(selectedDistrict);
        if (wbResults.length > 0) {
          results = wbResults as any[];
        }
      }
      
      // Second priority: Check expanded hospital database
      if (results.length === 0) {
        const stateDistrictResults = getHospitalsByStateDistrict(selectedState, selectedDistrict);
        if (stateDistrictResults.length > 0) {
          results = stateDistrictResults;
        }
      }
      
      // Third priority: Check city-based results if city is selected
      if (results.length === 0 && selectedCity) {
        const cityResults = getHospitalsWithinRadius(selectedCity, 100); // 100 km radius
        results = cityResults;
      }

      // Add distance calculation (mock) and prioritize government hospitals
      const resultsWithDistance = results.map(hospital => ({
        ...hospital,
        distance: Math.floor(Math.random() * 50) + 1 // 1-50 km
      })).sort((a, b) => {
        // Sort by: 1) Government first, 2) Distance
        if (a.category === 'govt' && b.category !== 'govt') return -1;
        if (a.category !== 'govt' && b.category === 'govt') return 1;
        return (a.distance || 0) - (b.distance || 0);
      });

      setSearchResults(resultsWithDistance as any);
      setIsSearching(false);
      setHasSearched(true);
      
      if (results.length === 0) {
        setShowNoResults(true);
      }
    }, 1000);
  };

  const handleStateSelect = (state: string) => {
    setSelectedState(state);
    setSelectedDistrict('');
    setSearchQuery('');
  };

  const handleDistrictSelect = (district: string) => {
    setSelectedDistrict(district);
    setSearchQuery('');
  };

  const handleCitySelect = (city: string) => {
    setSelectedCity(city);
    setSearchQuery('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const openWebsite = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <main className="max-w-4xl mx-auto px-4 py-8 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-4xl mb-4 text-gray-900">{t.findHelp.title}</h1>
          <p className="text-xl text-gray-600 mb-8">
            {t.findHelp.subtitle}
          </p>
        </motion.div>

        {/* Search Bar with Dropdown Mode */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8"
        >
          <Card>
            <CardContent className="p-6">
              <div className="space-y-4">
                {/* State Selection */}
                <div>
                  <label className="block text-sm mb-2 text-gray-700">
                    {comprehensiveT.findHelp.selectState || 'Select State'}
                  </label>
                  <div className="flex gap-2">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                      <Input
                        placeholder="Type or select state..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyPress={handleKeyPress}
                        className="pl-10"
                        disabled={selectedState !== ''}
                      />
                    </div>
                    <select
                      value={selectedState}
                      onChange={(e) => handleStateSelect(e.target.value)}
                      className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                    >
                      <option value="">-- Select State --</option>
                      {filteredStates.map((state) => (
                        <option key={state} value={state}>
                          {state}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedState && (
                    <div className="mt-2 flex items-center gap-2">
                      <span className="inline-block px-3 py-1 bg-teal-100 text-teal-800 rounded-full text-sm">
                        {selectedState}
                      </span>
                      <button
                        onClick={() => {
                          setSelectedState('');
                          setSelectedDistrict('');
                        }}
                        className="text-xs text-red-600 hover:text-red-700"
                      >
                        Clear
                      </button>
                    </div>
                  )}
                </div>

                {/* District Selection */}
                {selectedState && (
                  <div>
                    <label className="block text-sm mb-2 text-gray-700">
                      {comprehensiveT.findHelp.selectDistrict || 'Select District'}
                    </label>
                    <div className="flex gap-2">
                      <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                        <Input
                          placeholder="Type or select district..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          onKeyPress={handleKeyPress}
                          className="pl-10"
                          disabled={selectedDistrict !== ''}
                        />
                      </div>
                      <select
                        value={selectedDistrict}
                        onChange={(e) => handleDistrictSelect(e.target.value)}
                        className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                      >
                        <option value="">-- Select District --</option>
                        {filteredDistricts.map((district) => (
                          <option key={district} value={district}>
                            {district}
                          </option>
                        ))}
                      </select>
                    </div>
                    {selectedDistrict && (
                      <div className="mt-2 flex items-center gap-2">
                        <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                          {selectedDistrict}
                        </span>
                        <button
                          onClick={() => {
                            setSelectedDistrict('');
                          }}
                          className="text-xs text-red-600 hover:text-red-700"
                        >
                          Clear
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* City Selection */}
                <div>
                  <label className="block text-sm mb-2 text-gray-700">
                    {comprehensiveT.findHelp.selectCity || 'Select City'}
                  </label>
                  <div className="flex gap-2">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                      <Input
                        placeholder="Type or select city..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyPress={handleKeyPress}
                        className="pl-10"
                        disabled={selectedCity !== ''}
                      />
                    </div>
                    <select
                      value={selectedCity}
                      onChange={(e) => handleCitySelect(e.target.value)}
                      className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                    >
                      <option value="">-- Select City --</option>
                      {filteredCities.map((city) => (
                        <option key={city} value={city}>
                          {city}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedCity && (
                    <div className="mt-2 flex items-center gap-2">
                      <span className="inline-block px-3 py-1 bg-teal-100 text-teal-800 rounded-full text-sm">
                        {selectedCity}
                      </span>
                      <button
                        onClick={() => {
                          setSelectedCity('');
                        }}
                        className="text-xs text-red-600 hover:text-red-700"
                      >
                        Clear
                      </button>
                    </div>
                  )}
                </div>

                {/* Search Button */}
                <Button 
                  size="lg" 
                  className="w-full bg-teal-600 hover:bg-teal-700 py-6" 
                  onClick={handleSearch}
                  disabled={!selectedState || !selectedDistrict}
                >
                  {isSearching ? t.findHelp.searching : t.findHelp.searchButton}
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* No Results Message */}
        {showNoResults && !hasSearched && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8"
          >
            <Card className="border-2 border-red-200 bg-red-50">
              <CardContent className="p-6 text-center">
                <p className="text-red-900">
                  Please select at least a State and District to search for healthcare facilities.
                </p>
                <p className="text-sm text-red-700 mt-2">
                  Only Indian states, districts, and cities are supported.
                </p>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Government TB Programs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-8"
        >
          <h2 className="text-2xl mb-4 text-gray-900">{comprehensiveT.findHelp.govTitle}</h2>
          <div className="space-y-4">
            <Card className="border-teal-100">
              <CardContent className="p-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="text-lg mb-2 text-gray-900">{comprehensiveT.findHelp.ntepTitle}</h3>
                    <p className="text-gray-600 mb-3">
                      {comprehensiveT.findHelp.ntepDesc}
                    </p>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <Phone className="h-4 w-4" />
                      <span>{comprehensiveT.findHelp.ntepHelpline}</span>
                    </div>
                  </div>
                  <a 
                    href="https://dghs.mohfw.gov.in/national-tuberculosis-elimination-programme.php" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    <Button variant="outline" size="sm" className="flex-shrink-0">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      {comprehensiveT.findHelp.visitButton}
                    </Button>
                  </a>
                </div>
              </CardContent>
            </Card>

            <Card className="border-blue-100">
              <CardContent className="p-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="text-lg mb-2 text-gray-900">{comprehensiveT.findHelp.nikshayTitle}</h3>
                    <p className="text-gray-600 mb-3">
                      {comprehensiveT.findHelp.nikshayDesc}
                    </p>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <MapPin className="h-4 w-4" />
                      <span>{comprehensiveT.findHelp.nikshaySubtitle}</span>
                    </div>
                  </div>
                  <a 
                    href="https://nikshay.in/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    <Button variant="outline" size="sm" className="flex-shrink-0">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      {comprehensiveT.findHelp.visitButton}
                    </Button>
                  </a>
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>

        {/* Emergency Warning Signs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mb-8"
        >
          <Card className="border-2 border-red-200 bg-red-50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-red-900">
                <AlertTriangle className="h-6 w-6" />
                {comprehensiveT.findHelp.emergencyTitle}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-red-900">
              <ul className="space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  <span>{comprehensiveT.findHelp.emergency1}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  <span>{comprehensiveT.findHelp.emergency2}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  <span>{comprehensiveT.findHelp.emergency3}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  <span>{comprehensiveT.findHelp.emergency4}</span>
                </li>
              </ul>
              <div className="mt-4 p-3 bg-white rounded border border-red-200">
                <p className="text-sm">
                  <strong>{comprehensiveT.findHelp.emergencyNote}</strong> {comprehensiveT.findHelp.emergencyAction}
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Search Results */}
        {hasSearched && searchResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-8"
          >
            <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 mb-4">
              <h2 className="text-2xl text-teal-900">
                Healthcare Facilities in {selectedDistrict}, {selectedState}
              </h2>
              <p className="text-sm text-teal-700 mt-1">
                Found {searchResults.length} facilities • Sorted by distance (closest first)
              </p>
            </div>
            
            <div className="space-y-4">
              {searchResults.map((facility: any, index) => (
                <Card key={index} className={`border-2 ${
                  facility.category === 'govt' ? 'border-teal-200 bg-teal-50' : 'border-blue-200 bg-blue-50'
                }`}>
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-1 flex-wrap">
                          <h3 className="text-lg text-gray-900">{facility.name}</h3>
                          <span className={`px-3 py-1 rounded-full text-xs ${
                            facility.category === 'govt' 
                              ? 'bg-teal-600 text-white' 
                              : 'bg-blue-600 text-white'
                          }`}>
                            {facility.type}
                          </span>
                        </div>
                        {facility.distance && (
                          <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                            <NavigationIcon className="h-4 w-4 text-teal-600" />
                            <span className="font-semibold text-teal-700">{facility.distance.toFixed(1)} km away</span>
                          </div>
                        )}
                      </div>
                      {facility.website && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openWebsite(facility.website)}
                          className="flex-shrink-0"
                        >
                          <Globe className="h-4 w-4 mr-2" />
                          Website
                        </Button>
                      )}
                    </div>
                    
                    <div className="space-y-2 text-sm text-gray-700 mb-4">
                      <p className="flex items-start gap-2">
                        <MapPin className="h-4 w-4 flex-shrink-0 mt-0.5 text-gray-500" />
                        <span><strong>Location:</strong> {facility.location}</span>
                      </p>
                      <p className="flex items-start gap-2 ml-6">
                        <span className="text-gray-600">{facility.address}</span>
                      </p>
                      <p className="flex items-center gap-2">
                        <Phone className="h-4 w-4 text-gray-500" />
                        <span><strong>Phone:</strong> {facility.phone}</span>
                      </p>
                      {facility.emergencyPhone && (
                        <p className="flex items-center gap-2">
                          <Phone className="h-4 w-4 text-red-500" />
                          <span><strong>Emergency:</strong> {facility.emergencyPhone}</span>
                        </p>
                      )}
                      <p className="flex items-center gap-2">
                        <span className="font-medium">{comprehensiveT.findHelp.hoursLabel}</span>
                        <span>{facility.hours}</span>
                      </p>
                    </div>
                    
                    <div className={`border rounded p-3 ${
                      facility.category === 'govt' 
                        ? 'bg-white border-teal-200' 
                        : 'bg-white border-blue-200'
                    }`}>
                      <p className="text-sm text-gray-900">
                        <strong>{comprehensiveT.findHelp.servicesLabel}</strong> {facility.services}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </motion.div>
        )}

        {/* No Facilities Found */}
        {hasSearched && searchResults.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-8"
          >
            <Card className="border-2 border-gray-200">
              <CardContent className="p-8 text-center">
                <p className="text-gray-700 mb-2">
                  No healthcare facilities found in {selectedDistrict}, {selectedState}.
                </p>
                <p className="text-sm text-gray-600">
                  Please try selecting a different location or contact the NTEP helpline at 1800-11-6666 for assistance.
                </p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </main>
    </div>
  );
}