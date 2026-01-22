import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import { Label } from '../components/ui/label';
import { ImageWithFallback } from '../components/figma/ImageWithFallback';
import { 
  ArrowLeft, 
  ArrowRight, 
  MapPin
} from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { allIndianStates, getDistrictsByStateName } from '../data/comprehensive-indian-locations';

// Import provided symptom images
import feverImage from '../../assets/fever.png';
import coughingBloodImage from '../../assets/coughingblood.png';
import sputumBloodImage from '../../assets/sputumblood.png';
import nightSweatsImage from '../../assets/nightsweat.png';
import chestPainImage from '../../assets/chestpain.png';
import shortnessBreathImage from '../../assets/shortnessofbreath.png';
import weightLossImage from '../../assets/weightloss.png';
import fatigueWeaknessImage from '../../assets/tired.png';
import lumpsNeckImage from '../../assets/lumpsneck.png';
import coughTwoFourWeeksImage from '../../assets/persistentcough.png';
import swollenLymphNodeImage from '../../assets/swollenlymphnode.png';
import lossOfAppetiteImage from '../../assets/lossofappetite.png';

interface Question {
  id: string;
  imageUrl: string;
  type: 'boolean' | 'text';
}

const symptomImages = [
  feverImage,
  coughingBloodImage,
  sputumBloodImage,
  nightSweatsImage,
  chestPainImage,
  shortnessBreathImage,
  weightLossImage,
  fatigueWeaknessImage,
  lumpsNeckImage,
  coughTwoFourWeeksImage,
  swollenLymphNodeImage,
  lossOfAppetiteImage
];

const questionIds = [
  'feverTwoWeeks',
  'coughingBlood',
  'sputumBlood',
  'nightSweats',
  'chestPainBreathing',
  'shortnessOfBreath',
  'weightLoss',
  'fatigueWeakness',
  'lumpsArmpitsNeck',
  'coughTwoFourWeeks',
  'swollenLymphNode',
  'lossOfAppetite'
];

export default function AssessmentPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, boolean | string>>({});
  const [location, setLocation] = useState({ district: '', state: '' });
  const [showLocation, setShowLocation] = useState(false);
  const [direction, setDirection] = useState(1);
  const { language } = useLanguage();
  const t = getTranslation(language);

  const questions: Question[] = questionIds.map((id, index) => ({
    id,
    imageUrl: symptomImages[index],
    type: 'boolean' as const
  }));

  const totalSteps = questions.length + 1; // +1 for location
  const progress = ((currentStep + 1) / totalSteps) * 100;

  const handleAnswer = (answer: boolean) => {
    const currentQuestion = questions[currentStep];
    setAnswers(prev => ({ ...prev, [currentQuestion.id]: answer }));
    
    setTimeout(() => {
      handleNext();
    }, 300);
  };

  const handleNext = () => {
    if (currentStep < questions.length - 1) {
      setDirection(1);
      setCurrentStep(prev => prev + 1);
    } else if (currentStep === questions.length - 1) {
      setShowLocation(true);
      setDirection(1);
      setCurrentStep(prev => prev + 1);
    }
  };

  const handleBack = () => {
    if (showLocation) {
      setShowLocation(false);
      setDirection(-1);
      setCurrentStep(prev => prev - 1);
    } else if (currentStep > 0) {
      setDirection(-1);
      setCurrentStep(prev => prev - 1);
    } else {
      navigate('/');
    }
  };

  const handleSubmit = () => {
    if (!location.district || !location.state) {
      return;
    }

    const symptomCount = Object.values(answers).filter(v => v === true).length;
    
    let riskLevel = 'Low';
    let confidence = 30 + symptomCount * 5;
    
    if (symptomCount >= 5) {
      riskLevel = 'High';
      confidence = Math.min(75 + symptomCount * 3, 95);
    } else if (symptomCount >= 3) {
      riskLevel = 'Medium';
      confidence = 50 + symptomCount * 5;
    }

    navigate('/results', {
      state: {
        riskLevel,
        confidence,
        symptoms: answers,
        district: location.district,
        state: location.state
      }
    });
  };

  const currentQuestion = questions[currentStep];

  return (
    <div className="min-h-screen bg-gradient-to-b from-teal-50/30 to-white flex flex-col">
      <Navigation />

      <main className="flex-1 flex flex-col">
        {/* Progress Bar */}
        <div className="sticky top-16 bg-white border-b border-gray-200 z-40">
          <div className="max-w-3xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">
                Step {currentStep + 1} of {totalSteps}
              </span>
              <span className="text-sm text-teal-600">
                {Math.round(progress)}%
              </span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </div>

        {/* Question Content */}
        <div className="flex-1 flex items-center justify-center px-4 py-8">
          <div className="w-full max-w-2xl">
            <AnimatePresence mode="wait" custom={direction}>
              {!showLocation ? (
                <motion.div
                  key={currentStep}
                  custom={direction}
                  initial={{ opacity: 0, x: direction * 100 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: direction * -100 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="border-2 border-teal-100">
                    <CardContent className="p-8 sm:p-12">
                      {/* Symptom Image */}
                      {currentQuestion && (
                        <>
                          <div className="w-32 h-32 sm:w-40 sm:h-40 rounded-full overflow-hidden mx-auto mb-6 border-4 border-teal-100 shadow-lg">
                            <ImageWithFallback
                              src={currentQuestion.imageUrl}
                              alt={currentQuestion.id}
                              className="w-full h-full object-cover"
                            />
                          </div>

                          {/* Question */}
                          <h2 className="text-2xl sm:text-3xl mb-4 text-center text-gray-900">
                            {t.assessment.questions[currentStep].question}
                          </h2>
                          <p className="text-center text-gray-600 mb-10">
                            {t.assessment.questions[currentStep].description}
                          </p>

                          {/* Answer Buttons */}
                          <div className="space-y-3">
                            <div className="grid grid-cols-2 gap-4">
                              <Button
                                size="lg"
                                variant="outline"
                                onClick={() => handleAnswer(false)}
                                className={`py-8 text-lg rounded-xl border-2 transition-all ${
                                  answers[currentQuestion.id] === false
                                    ? 'border-red-400 bg-red-50 text-red-700'
                                    : 'border-gray-200 hover:border-red-300'
                                }`}
                              >
                                {t.assessment.buttons.no}
                              </Button>
                              <Button
                                size="lg"
                                onClick={() => handleAnswer(true)}
                                className={`py-8 text-lg rounded-xl transition-all ${
                                  answers[currentQuestion.id] === true
                                    ? 'bg-teal-600 hover:bg-teal-700'
                                    : 'bg-teal-500 hover:bg-teal-600'
                                }`}
                              >
                                {t.assessment.buttons.yes}
                              </Button>
                            </div>
                            
                            {/* Back Button */}
                            <Button
                              size="lg"
                              variant="outline"
                              onClick={handleBack}
                              className="w-full py-6 text-lg rounded-xl border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                            >
                              <ArrowLeft className="mr-2 h-5 w-5" />
                              {t.assessment.buttons.back}
                            </Button>
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              ) : (
                <motion.div
                  key="location"
                  custom={direction}
                  initial={{ opacity: 0, x: direction * 100 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: direction * -100 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="border-2 border-teal-100">
                    <CardContent className="p-8 sm:p-12">
                      {/* Icon */}
                      <div className="w-20 h-20 bg-teal-100 rounded-full flex items-center justify-center mx-auto mb-6">
                        <MapPin className="h-10 w-10 text-teal-600" />
                      </div>

                      {/* Question */}
                      <h2 className="text-2xl sm:text-3xl mb-4 text-center text-gray-900">
                        {t.assessment.location.title}
                      </h2>
                      <p className="text-center text-gray-600 mb-10">
                        {t.assessment.location.description}
                      </p>

                      {/* Location Inputs */}
                      <div className="space-y-6">
                        {/* State Dropdown - First */}
                        <div>
                          <Label htmlFor="state" className="text-base mb-2 block">
                            {t.assessment.location.state}
                          </Label>
                          <select
                            id="state"
                            value={location.state}
                            onChange={(e) => setLocation({ state: e.target.value, district: '' })}
                            className="w-full text-lg py-3 px-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                          >
                            <option value="">{t.assessment.location.statePlaceholder}</option>
                            {allIndianStates.map((state) => (
                              <option key={state.name} value={state.name}>
                                {state.name}
                              </option>
                            ))}
                          </select>
                        </div>

                        {/* District Dropdown - Second */}
                        <div>
                          <Label htmlFor="district" className="text-base mb-2 block">
                            {t.assessment.location.district}
                          </Label>
                          <select
                            id="district"
                            value={location.district}
                            onChange={(e) => setLocation(prev => ({ ...prev, district: e.target.value }))}
                            disabled={!location.state}
                            className={`w-full text-lg py-3 px-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 ${
                              !location.state ? 'bg-gray-100 cursor-not-allowed' : ''
                            }`}
                          >
                            <option value="">{t.assessment.location.districtPlaceholder}</option>
                            {location.state && getDistrictsByStateName(location.state).map((district) => (
                              <option key={district} value={district}>
                                {district}
                              </option>
                            ))}
                          </select>
                        </div>

                        <Button
                          size="lg"
                          onClick={handleSubmit}
                          disabled={!location.district || !location.state}
                          className="w-full py-7 text-lg bg-teal-600 hover:bg-teal-700 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {t.assessment.location.submit}
                          <ArrowRight className="ml-2 h-5 w-5" />
                        </Button>
                        
                        {/* Back Button */}
                        <Button
                          size="lg"
                          variant="outline"
                          onClick={handleBack}
                          className="w-full py-6 text-lg rounded-xl border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                        >
                          <ArrowLeft className="mr-2 h-5 w-5" />
                          {t.assessment.buttons.back}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}