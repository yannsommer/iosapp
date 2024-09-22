import SwiftUI
import Vision
import CoreML

struct ContentView: View {
    @State private var image: UIImage?
    @State private var detectionResults: [DetectionResult] = []
    @State private var isShowingImagePicker = false
    
    private let objectDetector = ObjectDetector()
    
    var body: some View {
        VStack {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .overlay(DetectionOverlay(detectionResults: detectionResults))
            } else {
                Text("No image selected")
            }
            
            Button("Select Image") {
                isShowingImagePicker = true
            }
            .padding()
            
            List(detectionResults) { result in
                Text("\(result.label): \(String(format: "%.2f", result.confidence))")
            }
        }
        .sheet(isPresented: $isShowingImagePicker) {
            ImagePicker(image: $image, detectionResults: $detectionResults, objectDetector: objectDetector)
        }
    }
}

struct DetectionResult: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

struct DetectionOverlay: View {
    let detectionResults: [DetectionResult]
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(detectionResults) { result in
                let rect = VNImageRectForNormalizedRect(result.boundingBox, Int(geometry.size.width), Int(geometry.size.height))
                Rectangle()
                    .path(in: rect)
                    .stroke(Color.red, lineWidth: 2)
            }
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Binding var detectionResults: [DetectionResult]
    let objectDetector: ObjectDetector
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
                if let cgImage = uiImage.cgImage {
                    parent.objectDetector.detect(image: cgImage) { observations in
                        DispatchQueue.main.async {
                            self.parent.detectionResults = observations.map { observation in
                                DetectionResult(
                                    label: observation.labels[0].identifier,
                                    confidence: observation.confidence,
                                    boundingBox: observation.boundingBox
                                )
                            }
                        }
                    }
                }
            }
            picker.dismiss(animated: true)
        }
    }
}

class ObjectDetector {
    private var visionModel: VNCoreMLModel
    
    init() {
        do {
            let config = MLModelConfiguration()
            guard let model = try? yolov5su(configuration: config) else {
                fatalError("Failed to load the ML model.")
            }
            visionModel = try VNCoreMLModel(for: model.model)
            print("Model loaded successfully")
        } catch {
            fatalError("Failed to create Vision model: \(error)")
        }
    }
    
    func detect(image: CGImage, completion: @escaping ([VNRecognizedObjectObservation]) -> Void) {
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                print("No results or wrong type returned")
                completion([])
                return
            }
            print("Number of detections: \(results.count)")
            for (index, result) in results.enumerated() {
                print("Detection \(index + 1): \(result.labels.first?.identifier ?? "Unknown"), confidence: \(result.confidence)")
            }
            completion(results)
        }
        
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform detection: \(error)")
            completion([])
        }
    }
}

        

