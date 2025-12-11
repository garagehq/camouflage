#!/usr/bin/env swift

import CoreLocation
import Foundation

class LocationManager: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private var completion: ((CLLocation?) -> Void)?

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
    }

    func getLocation(timeout: TimeInterval = 10, completion: @escaping (CLLocation?) -> Void) {
        self.completion = completion

        // Check authorization status
        let status = manager.authorizationStatus

        switch status {
        case .notDetermined:
            // Request authorization - this will prompt the user
            manager.requestWhenInUseAuthorization()
            // Start updating after a brief delay to allow authorization
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                self.manager.startUpdatingLocation()
            }
        case .authorizedAlways, .authorizedWhenInUse:
            manager.startUpdatingLocation()
        case .denied, .restricted:
            // Try to get last known location even if denied
            if let location = manager.location {
                completion(location)
                return
            }
            fputs("error: Location services denied or restricted\n", stderr)
            completion(nil)
            return
        @unknown default:
            completion(nil)
            return
        }

        // Timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
            if self.completion != nil {
                self.manager.stopUpdatingLocation()
                // Try last known location as fallback
                if let location = self.manager.location {
                    self.completion?(location)
                } else {
                    self.completion?(nil)
                }
                self.completion = nil
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        manager.stopUpdatingLocation()
        completion?(location)
        completion = nil
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        fputs("error: \(error.localizedDescription)\n", stderr)
        manager.stopUpdatingLocation()
        // Try last known location as fallback
        if let location = manager.location {
            completion?(location)
        } else {
            completion?(nil)
        }
        completion = nil
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        if status == .authorizedAlways || status == .authorizedWhenInUse {
            manager.startUpdatingLocation()
        }
    }
}

// Main
let locationManager = LocationManager()
let semaphore = DispatchSemaphore(value: 0)

locationManager.getLocation(timeout: 10) { location in
    if let loc = location {
        // Output as JSON for easy parsing
        print("{\"latitude\": \(loc.coordinate.latitude), \"longitude\": \(loc.coordinate.longitude), \"accuracy\": \(loc.horizontalAccuracy)}")
    } else {
        fputs("error: Could not get location\n", stderr)
        exit(1)
    }
    semaphore.signal()
}

// Keep the run loop alive
RunLoop.main.run(until: Date(timeIntervalSinceNow: 12))
semaphore.wait()
