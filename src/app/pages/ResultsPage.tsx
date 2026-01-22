import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Alert,
  AlertDescription,
} from "../components/ui/alert";
import { Progress } from "../components/ui/progress";
import {
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  MapPin,
  Phone,
  Clock,
  ArrowRight,
  ExternalLink,
} from "lucide-react";
import Navigation from "../components/Navigation";
import { useLanguage } from "../context/LanguageContext";
import { getTranslation } from "../translations/translations";
import {
  getHospitalsByDistrict,
  getHospitalsByState,
  getNearbyStateHospitals,
} from "../data/hospitalDatabase";

interface ResultState {
  riskLevel: string;
  confidence: number;
  symptoms: Record<string, any>;
  district: string;
  state: string;
}

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { language } = useLanguage();
  const t = getTranslation(language);

  const state = location.state as ResultState;

  useEffect(() => {
    if (!state) {
      navigate("/");
    }
  }, [state, navigate]);

  if (!state) {
    return null;
  }

  const {
    riskLevel,
    confidence,
    district,
    state: userState,
  } = state;

  const getRiskColor = () => {
    switch (riskLevel) {
      case "High":
        return {
          bg: "bg-red-50",
          border: "border-red-300",
          text: "text-red-900",
          icon: "text-red-600",
          button: "bg-red-600 hover:bg-red-700",
        };
      case "Medium":
        return {
          bg: "bg-amber-50",
          border: "border-amber-300",
          text: "text-amber-900",
          icon: "text-amber-600",
          button: "bg-amber-600 hover:bg-amber-700",
        };
      default:
        return {
          bg: "bg-emerald-50",
          border: "border-emerald-300",
          text: "text-emerald-900",
          icon: "text-emerald-600",
          button: "bg-emerald-600 hover:bg-emerald-700",
        };
    }
  };

  const getRiskIcon = () => {
    switch (riskLevel) {
      case "High":
        return <AlertTriangle className="h-16 w-16" />;
      case "Medium":
        return <AlertCircle className="h-16 w-16" />;
      default:
        return <CheckCircle2 className="h-16 w-16" />;
    }
  };

  const getRiskMessage = () => {
    switch (riskLevel) {
      case "High":
        return t.results.highDesc;
      case "Medium":
        return t.results.mediumDesc;
      default:
        return t.results.lowDesc;
    }
  };

  const getRiskLabel = () => {
    switch (riskLevel) {
      case "High":
        return t.results.high;
      case "Medium":
        return t.results.medium;
      default:
        return t.results.low;
    }
  };

  const colors = getRiskColor();
  const riskMessage = getRiskMessage();
  const riskLabel = getRiskLabel();
  const showHospitals =
    riskLevel === "High" || riskLevel === "Medium";

  // Get hospitals based on user location
  let hospitals = getHospitalsByDistrict(district, userState);
  let locationLabel = `${district}, ${userState}`;

  if (hospitals.length === 0) {
    hospitals = getHospitalsByState(userState);
    locationLabel = userState;
  }

  if (hospitals.length === 0) {
    hospitals = getNearbyStateHospitals(userState);
    locationLabel = `nearby locations`;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <Navigation />

      <main className="max-w-3xl mx-auto px-4 py-8 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-6"
        >
          <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-center">
            <p className="text-sm text-blue-900">
              <strong>{t.results.disclaimer}:</strong>{" "}
              {t.results.disclaimerText}
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <Card
            className={`${colors.border} ${colors.bg} border-2`}
          >
            <CardContent className="p-6 sm:p-10">
              <div className="text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{
                    duration: 0.5,
                    delay: 0.2,
                    type: "spring",
                  }}
                  className={`inline-flex items-center justify-center w-20 h-20 sm:w-24 sm:h-24 ${colors.bg} rounded-full ${colors.icon} mb-6`}
                >
                  {getRiskIcon()}
                </motion.div>

                <h1
                  className={`text-2xl sm:text-3xl mb-3 ${colors.text}`}
                >
                  {riskLabel}
                </h1>

                <div className="max-w-md mx-auto mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">
                      Confidence
                    </span>
                    <span className={`text-sm ${colors.text}`}>
                      {confidence}%
                    </span>
                  </div>
                  <Progress
                    value={confidence}
                    className="h-2"
                  />
                </div>

                <p className={`text-lg mb-6 ${colors.text}`}>
                  {riskMessage}
                </p>

                <Alert
                  className={`${colors.border} ${colors.bg} text-left`}
                >
                  <AlertCircle
                    className={`h-4 w-4 ${colors.icon}`}
                  />
                  <AlertDescription className={colors.text}>
                    <strong>Recommended action:</strong>{" "}
                    {riskLevel === "High" ? t.results.highAction : riskLevel === "Medium" ? t.results.mediumAction : t.results.lowAction}
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {showHospitals && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mb-8"
          >
            <Button
              size="lg"
              onClick={() => navigate("/find-help")}
              className={`w-full py-7 text-lg rounded-xl ${colors.button}`}
            >
              Find Nearby Clinic
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </motion.div>
        )}

        {showHospitals && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mb-8"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <MapPin className="h-5 w-5 text-teal-600" />
                  Nearby Healthcare Facilities in{" "}
                  {locationLabel}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {hospitals.map((hospital, index) => (
                    <div
                      key={index}
                      className="p-4 border border-gray-200 rounded-lg hover:border-teal-300 transition-colors"
                    >
                      <h3 className="mb-2 text-gray-900">
                        {hospital.name}
                      </h3>
                      <div className="space-y-1 text-sm text-gray-600 mb-3">
                        <p className="flex items-center gap-2">
                          <MapPin className="h-4 w-4 text-gray-400" />
                          {hospital.address}
                        </p>
                        <p className="flex items-center gap-2">
                          <Phone className="h-4 w-4 text-gray-400" />
                          {hospital.phone}
                        </p>
                        <p className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-gray-400" />
                          {hospital.hours}
                        </p>
                      </div>
                      {hospital.website && (
                        <a
                          href={hospital.website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 text-teal-600 hover:text-teal-700 text-sm"
                        >
                          Visit Website
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mb-8"
        >
          <Card className="border-amber-200 bg-amber-50/50">
            <CardContent className="p-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-900">
                  <p className="mb-2">
                    <strong>
                      This is NOT a medical diagnosis.
                    </strong>{" "}
                    Only a qualified healthcare professional can
                    diagnose TB through clinical examination and
                    tests.
                  </p>
                  <p>
                    If you have concerns about TB or any health
                    condition, please consult a doctor
                    regardless of this result.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="text-center"
        >
          <Button
            variant="outline"
            onClick={() => navigate("/assessment")}
            className="border-gray-300 text-gray-700"
          >
            Take Assessment Again
          </Button>
        </motion.div>
      </main>
    </div>
  );
}