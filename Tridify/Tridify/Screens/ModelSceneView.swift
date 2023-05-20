//
//  ModelSceneView.swift
//  Tridify
//
//  Created by Maged Alosali on 19/05/2023.
//

import SwiftUI
import SceneKit

struct ModelSceneView: View {
    
    private let path = URL(fileURLWithPath: Bundle.main.path(forResource: "DeathBottle_Bottle", ofType: "stl")!)
    private let modelName: String?
    @State private var scene: SCNScene?
    @State private var error: Error?
    var body: some View {
        VStack {
            if let scene = scene {
                SceneView(scene: scene, options: [.autoenablesDefaultLighting, .allowsCameraControl])

            }
            else if let error = error {
                Text("Error: \(error.localizedDescription)")
            }
            else {
                ProgressView()
            }
        }
        .navigationTitle(modelName ?? "")
        .onAppear {
            do {
                let load = try SCNScene(url: path)
                scene = load
                scene?.background.contents =  UIColor.gray.withAlphaComponent(0.2)
                
            } catch {
                self.error = error
            }
    
        }
    }
    
    init(modelName: String?) {
        self.modelName = modelName
    }
}

struct ModelSceneView_Previews: PreviewProvider {
    static var previews: some View {
        
        ModelSceneView(modelName: "Perfume")
    }
}
